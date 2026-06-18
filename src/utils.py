import os
import yaml
import shutil


def load_config(path="config.yaml"):
    """Load YAML config and return as dict."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_kaggle_credentials(kaggle_json_path=None):
    """
    Set up Kaggle API credentials for dataset download in Colab.

    Priority order:
    1. Explicit path to kaggle.json passed as argument
    2. Default Google Drive location: /content/drive/MyDrive/kaggle.json
    3. Already set via environment variables KAGGLE_USERNAME / KAGGLE_KEY

    Args:
        kaggle_json_path: optional direct path to kaggle.json
    """
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        print("Kaggle credentials already set via environment variables.")
        return

    if kaggle_json_path is None:
        kaggle_json_path = '/content/drive/MyDrive/kaggle.json'

    if os.path.exists(kaggle_json_path):
        dest = os.path.expanduser('~/.kaggle/kaggle.json')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(kaggle_json_path, dest)
        os.chmod(dest, 0o600)
        print(f"Kaggle credentials loaded from {kaggle_json_path}")
    else:
        raise FileNotFoundError(
            f"kaggle.json not found at {kaggle_json_path}.\n"
            "Upload it to Google Drive at MyDrive/kaggle.json, or set "
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )


def download_datasets(cfg):
    """
    Download LOL and DarkFace datasets via kagglehub.

    Args:
        cfg: loaded config dict (uses cfg['datasets'])

    Returns:
        lol_path (str), darkface_path (str)
    """
    import kagglehub

    print("Downloading LOL dataset...")
    lol_raw = kagglehub.dataset_download(cfg['datasets']['lol']['kaggle_id'])
    print(f"  LOL raw path: {lol_raw}")

    print("Downloading DarkFace dataset...")
    darkface_raw = kagglehub.dataset_download(cfg['datasets']['darkface']['kaggle_id'])
    print(f"  DarkFace raw path: {darkface_raw}")

    # Copy to stable local paths
    lol_dest      = '/content/dataset/lol_dataset'
    darkface_dest = '/content/darkface_dataset'

    if not os.path.exists(lol_dest):
        print(f"Copying LOL dataset to {lol_dest}...")
        shutil.copytree(lol_raw, lol_dest, dirs_exist_ok=True)

    if not os.path.exists(darkface_dest):
        print(f"Copying DarkFace dataset to {darkface_dest}...")
        shutil.copytree(darkface_raw, darkface_dest, dirs_exist_ok=True)

    print("Datasets ready.")
    return lol_dest, darkface_dest


def make_drive_dirs(cfg):
    """Create output directories on Google Drive."""
    for key in ['checkpoints', 'results']:
        path = cfg['paths'][key]
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")


def get_device():
    """Return best available device and print info."""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        import torch.cuda as cuda
        print(f"  GPU: {cuda.get_device_name(0)}")
        print(f"  Memory: {cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def preflight_check(root, img_dir='image', lbl_dir='label'):
    """Quick directory existence checks for DarkFace dataset layout."""
    imgp = os.path.join(root, img_dir)
    lblp = os.path.join(root, lbl_dir)
    print("=== Preflight check ===")
    print(("OK" if os.path.isdir(root)  else "MISSING"), "root :", root)
    print(("OK" if os.path.isdir(imgp)  else "MISSING"), "images:", imgp)
    print(("OK" if os.path.isdir(lblp)  else "MISSING"), "labels:", lblp)
    if os.path.isdir(imgp):
        imgs = [f for f in os.listdir(imgp) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  image count: {len(imgs)}")
    print("=======================")
