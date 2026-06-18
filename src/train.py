import os
import time
import math
import csv
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_fn


# ─────────────────────────────────────────────
# LOL training (supervised — paired images)
# ─────────────────────────────────────────────

def train_lol(model, train_loader, val_loader, device, cfg, save_dir):
    """
    Supervised training on LOL dataset using paired low/high images.
    - 200 epochs
    - Adam optimizer, lr=1e-3
    - StepLR: lr halved every 5 epochs
    - MSE loss for training
    - PSNR + SSIM for evaluation
    Saves best_model.pth and final_model.pth to save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses, learning_rates = [], [], []
    val_psnrs, val_ssims = [], []
    best_val_loss = float('inf')

    for epoch in range(cfg['epochs']):
        t0 = time.time()

        # ── Train ──
        model.train()
        running = 0.0
        for low_imgs, high_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False):
            low_imgs  = low_imgs.to(device)
            high_imgs = high_imgs.to(device)

            enhanced = model(low_imgs)
            loss     = criterion(enhanced, high_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg_train = running / max(1, len(train_loader))

        # ── Validate ──
        model.eval()
        val_running  = 0.0
        ssim_running = 0.0
        n_batches    = 0
        with torch.no_grad():
            for low_imgs, high_imgs in val_loader:
                low_imgs  = low_imgs.to(device)
                high_imgs = high_imgs.to(device)
                enhanced  = model(low_imgs)
                # MSE loss
                val_running += criterion(enhanced, high_imgs).item()
                # SSIM — expects values in [0, 1]
                # tanh output is [-1,1] so rescale to [0,1] first
                enh_01  = (enhanced  + 1.0) / 2.0
                high_01 = (high_imgs + 1.0) / 2.0
                ssim_running += ssim_fn(enh_01, high_01,
                                        data_range=1.0,
                                        size_average=True).item()
                n_batches += 1
        avg_val  = val_running  / max(1, len(val_loader))
        avg_ssim = ssim_running / max(1, n_batches)
        avg_psnr = -10.0 * math.log10(max(avg_val, 1e-12))
        # StepLR steps every epoch automatically
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        learning_rates.append(current_lr)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        print(f"   Train Loss : {avg_train:.6f}")
        print(f"   Val Loss   : {avg_val:.6f}")
        print(f"   PSNR       : {avg_psnr:.2f} dB")
        print(f"   SSIM       : {avg_ssim:.4f}")
        print(f"   LR         : {current_lr:.2e}")
        print(f"   Time       : {time.time()-t0:.1f}s")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"   Saved best model (val loss: {best_val_loss:.6f})")

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"\nSaved final_model.pth to {save_dir}")

    return train_losses, val_losses, learning_rates, val_psnrs, val_ssims


# ─────────────────────────────────────────────
# DarkFace training (self-supervised)
# ─────────────────────────────────────────────

def train_darkface(model, train_loader, test_loader, device, cfg, save_dir,
                   vis_interval=5, metrics_log=None):
    """
    Self-supervised training on DarkFace dataset (no paired ground truth).
    Model learns to enhance its own input (output should look better than input).

    Args:
        model:         HybridEnhancementNet instance
        train_loader:  DataLoader for DarkFace training split
        test_loader:   DataLoader for DarkFace test split (or None)
        device:        torch.device
        cfg:           dict with keys: epochs, lr, step_size, gamma
        save_dir:      path to save checkpoints
        vis_interval:  visualize every N epochs
        metrics_log:   list to append PSNR/MSE dicts to
    """
    os.makedirs(save_dir, exist_ok=True)

    if metrics_log is None:
        metrics_log = []

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg['step_size'], gamma=cfg['gamma']
    )

    train_losses, learning_rates = [], []

    for epoch in range(cfg['epochs']):
        t0 = time.time()
        model.train()
        running = 0.0

        for low_imgs, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False):
            low_imgs = low_imgs.to(device)
            outputs  = model(low_imgs)
            loss     = criterion(outputs, low_imgs)   # self-supervised: output vs input

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg_train = running / max(1, len(train_loader))
        train_losses.append(avg_train)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        print(f"   Train Loss : {avg_train:.6f}")
        print(f"   LR         : {current_lr:.2e}")
        print(f"   Time       : {time.time()-t0:.1f}s")

        if (epoch + 1) % vis_interval == 0:
            if test_loader is not None and len(test_loader) > 0:
                _evaluate_and_log(model, test_loader, device, criterion,
                                  split='test', epoch=epoch+1, log=metrics_log)

    torch.save(model.state_dict(), os.path.join(save_dir, 'best_enhancement_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_enhancement_model.pth'))
    print(f"\nSaved enhancement model checkpoints to {save_dir}")

    return train_losses, learning_rates


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _evaluate_and_log(model, dataloader, device, criterion, split, epoch, log):
    model.eval()
    mse_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            # Support both (low, high) and (low, boxes, sizes) formats
            low_imgs = batch[0].to(device)
            outputs  = model(low_imgs)
            mse_per  = torch.mean((outputs - low_imgs)**2, dim=[1, 2, 3])
            mse_sum += mse_per.sum().item()
            n       += low_imgs.size(0)
    avg_mse  = mse_sum / max(1, n)
    psnr_db  = -10.0 * math.log10(max(avg_mse, 1e-12))
    log.append({'epoch': epoch, 'split': split, 'samples': n,
                'mse': avg_mse, 'psnr_db': psnr_db})
    print(f"   [{split}] MSE: {avg_mse:.6f}  PSNR: {psnr_db:.2f} dB")
    return avg_mse, psnr_db


def save_metrics_csv(metrics_log, path):
    if not metrics_log:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'split', 'samples', 'mse', 'psnr_db'])
        writer.writeheader()
        writer.writerows(metrics_log)
    print(f"Saved metrics to {path}")


def print_metrics_table(metrics_log):
    if not metrics_log:
        print("No metrics logged.")
        return
    print("\n| Epoch | Split  | Samples |      MSE      | PSNR (dB) |")
    print("|------:|:-------|--------:|--------------:|----------:|")
    for r in metrics_log:
        print(f"| {r['epoch']:>5} | {r['split']:<6} | {r['samples']:>7} | {r['mse']:>13.6f} | {r['psnr_db']:>9.2f} |")
