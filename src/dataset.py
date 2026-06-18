import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class LOLDataset(Dataset):
    """
    Dataset for the LOL (Low-Light) paired dataset.
    Source: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

    Folder structure expected:
        root_dir/
            our485/
                low/    <- low-light training images
                high/   <- normal-light training images
            eval15/
                low/    <- low-light test images
                high/   <- normal-light test images

    Args:
        root_dir: path to lol_dataset folder
        transform: torchvision transforms to apply
        train: if True loads our485/, else loads eval15/
    """

    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        folder = 'our485' if train else 'eval15'

        low_dir  = os.path.join(root_dir, folder, 'low')
        high_dir = os.path.join(root_dir, folder, 'high')

        if not os.path.isdir(low_dir):
            raise FileNotFoundError(f"Low-light folder not found: {low_dir}")
        if not os.path.isdir(high_dir):
            raise FileNotFoundError(f"High-light folder not found: {high_dir}")

        self.low_light_images  = sorted(os.listdir(low_dir))
        self.high_light_images = sorted(os.listdir(high_dir))
        self.folder = folder

        print(f"LOLDataset ({'train' if train else 'test'}): {len(self.low_light_images)} pairs")

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        low_path  = os.path.join(self.root_dir, self.folder, 'low',  self.low_light_images[idx])
        high_path = os.path.join(self.root_dir, self.folder, 'high', self.high_light_images[idx])

        low_img  = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        if self.transform:
            low_img  = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


class DarkFaceDataset(Dataset):
    """
    Dataset for DarkFace detection dataset.
    Source: https://www.kaggle.com/datasets/soumikrakshit/dark-face-dataset

    Folder structure expected:
        root_dir/
            image/   <- dark face images (.jpg/.png)
            label/   <- annotation .txt files (count on line1, then xmin ymin xmax ymax per line)

    Args:
        root_dir: path to darkface_dataset folder
        transform: torchvision transforms to apply
        images_dir: subfolder name for images (default: 'image')
        labels_dir: subfolder name for labels (default: 'label')
    """

    def __init__(self, root_dir, transform=None,
                 images_dir='image', labels_dir='label'):
        self.root_dir   = root_dir
        self.transform  = transform
        self.images_dir = os.path.join(root_dir, images_dir)
        self.labels_dir = os.path.join(root_dir, labels_dir)

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root dir not found: {root_dir}")
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"'image' folder not found: {self.images_dir}")
        if not os.path.isdir(self.labels_dir):
            print(f"Warning: 'label' folder not found at {self.labels_dir} — continuing without labels")

        self.image_paths = []
        self.annotations = []
        self._load_dataset()

        print(f"DarkFaceDataset: {len(self.image_paths)} images loaded")

    def _load_dataset(self):
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        for f in sorted(os.listdir(self.images_dir)):
            if f.lower().endswith(exts):
                self.image_paths.append(os.path.join(self.images_dir, f))

        for img_path in self.image_paths:
            stem     = os.path.splitext(os.path.basename(img_path))[0]
            ann_path = os.path.join(self.labels_dir, f"{stem}.txt")
            boxes    = self._parse_annotation(ann_path) if os.path.exists(ann_path) else []
            self.annotations.append(boxes)

    def _parse_annotation(self, txt_path):
        boxes = []
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:   # skip count line
                data = line.strip().split()
                if len(data) >= 4:
                    xmin, ymin, xmax, ymax = float(data[0]), float(data[1]), float(data[2]), float(data[3])
                    boxes.append([xmin, ymin, xmax, ymax, 0])  # class 0 = face
        except Exception as e:
            print(f"Warning: failed to parse {txt_path}: {e}")
        return boxes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        boxes    = self.annotations[idx]

        try:
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
        except Exception as e:
            print(f"Warning: failed to read {img_path}: {e} — using synthetic image")
            img = Image.fromarray(np.random.randint(0, 100, (480, 640, 3), dtype=np.uint8))
            original_size = (640, 480)

        if self.transform:
            img = self.transform(img)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5), dtype=torch.float32)

        return img, boxes_tensor, original_size


def darkface_collate_fn(batch):
    """
    Custom collate for DarkFaceDataset — pads images to the max H and W in the batch.
    Needed because DarkFace images have variable sizes.
    """
    images, boxes, sizes = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded = []
    for img in images:
        ph = max_h - img.shape[1]
        pw = max_w - img.shape[2]
        padded.append(torch.nn.functional.pad(img, (0, pw, 0, ph)))
    return torch.stack(padded), boxes, sizes
