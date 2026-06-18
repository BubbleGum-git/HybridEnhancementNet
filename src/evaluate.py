import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def compute_psnr_mse(model, dataloader, device, paired=True):
    """
    Compute MSE and PSNR on a dataloader.

    Args:
        model:      trained HybridEnhancementNet
        dataloader: DataLoader (LOL paired or DarkFace self-supervised)
        device:     torch.device
        paired:     if True, compares enhanced vs high-light ground truth
                    if False (DarkFace), compares enhanced vs input (self-supervised)

    Returns:
        avg_mse (float), psnr_db (float)
    """
    model.eval()
    mse_sum, n = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            if paired:
                low_imgs, high_imgs = batch[0].to(device), batch[1].to(device)
                outputs  = model(low_imgs)
                target   = high_imgs
            else:
                low_imgs = batch[0].to(device)
                outputs  = model(low_imgs)
                target   = low_imgs

            mse_per  = torch.mean((outputs - target) ** 2, dim=[1, 2, 3])
            mse_sum += mse_per.sum().item()
            n       += low_imgs.size(0)

    avg_mse = mse_sum / max(1, n)
    psnr_db = -10.0 * math.log10(max(avg_mse, 1e-12))

    print(f"Test Results — MSE: {avg_mse:.6f}  PSNR: {psnr_db:.2f} dB")
    return avg_mse, psnr_db


def visualize_enhancement(model, dataloader, device, save_path, tag="results",
                           paired=True, n_show=4):
    """
    Save a grid image showing: low-light input | enhanced output | (ground truth if paired).

    Args:
        model:      trained HybridEnhancementNet
        dataloader: DataLoader to sample one batch from
        device:     torch.device
        save_path:  full path to save the .png file
        tag:        label shown in plot title
        paired:     if True, also shows ground truth column
        n_show:     number of images to show per row
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        low_imgs = batch[0].to(device)
        enhanced = model(low_imgs).cpu()
        low_imgs = low_imgs.cpu()

        if paired:
            high_imgs = batch[1].cpu()
            n_rows = 3
            grids  = [
                vutils.make_grid(low_imgs[:n_show],  nrow=n_show, normalize=True),
                vutils.make_grid(enhanced[:n_show],  nrow=n_show, normalize=True),
                vutils.make_grid(high_imgs[:n_show], nrow=n_show, normalize=True),
            ]
            titles = ['Low-light input', 'Enhanced output', 'Ground truth']
        else:
            n_rows = 2
            grids  = [
                vutils.make_grid(low_imgs[:n_show], nrow=n_show, normalize=True),
                vutils.make_grid(enhanced[:n_show], nrow=n_show, normalize=True),
            ]
            titles = ['Low-light input', 'Enhanced output']

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 5 * n_rows))
    fig.suptitle(f'Enhancement Results — {tag}', fontsize=16, fontweight='bold')

    for ax, grid, title in zip(axes, grids, titles):
        ax.imshow(grid.permute(1, 2, 0))
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_training_curves(train_losses, val_losses, learning_rates, save_path, title="Training History"):
    """
    Plot loss curves and LR schedule side by side.

    Args:
        train_losses:   list of per-epoch train losses
        val_losses:     list of per-epoch val losses (can be empty for self-supervised)
        learning_rates: list of per-epoch learning rates
        save_path:      full path to save the .png
        title:          plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Train loss', linewidth=2, color='steelblue')
    if val_losses:
        ax1.plot(epochs, val_losses, label='Val loss', linewidth=2, color='tomato')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Loss curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, learning_rates, label='Learning rate', linewidth=2, color='seagreen')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('LR')
    ax2.set_yscale('log')
    ax2.set_title('LR schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def visualize_dataset_samples(dataset, save_path, title="Dataset samples",
                               num_samples=5, paired=True):
    """
    Show random samples from a dataset side by side (low vs high for LOL).
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    n_rows  = 2 if paired else 1

    fig, axes = plt.subplots(n_rows, len(indices), figsize=(4 * len(indices), 4 * n_rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if len(indices) == 1:
        axes = axes.reshape(n_rows, 1)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        low_np = sample[0].permute(1, 2, 0).numpy()
        low_np = np.clip(low_np, 0, 1)
        axes[0, i].imshow(low_np)
        axes[0, i].set_title(f'Low {i+1}')
        axes[0, i].axis('off')

        if paired and len(sample) >= 2:
            high_np = sample[1].permute(1, 2, 0).numpy()
            high_np = np.clip(high_np, 0, 1)
            axes[1, i].imshow(high_np)
            axes[1, i].set_title(f'High {i+1}')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved {save_path}")
