"""
glas_dataset.py — GlaS Dataset Loader

Augmentation: stain norm, flips, 90° rotations, affine, color jitter, elastic deformation
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

try:
    import albumentations as A
    ALBUMENTATIONS = True
except ImportError:
    ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Elastic deformation disabled.")
    print("Install with: pip install albumentations --break-system-packages")


DATA_ROOT = Path("/opt/dlami/nvme/HTAN/data/glas")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"


def stain_normalize(image):
    arr = np.asarray(image).astype(np.float32)
    out = np.zeros_like(arr)
    for c in range(3):
        ch   = arr[:, :, c]
        low  = np.percentile(ch, 1)
        high = np.percentile(ch, 99)
        ch   = np.clip(ch, low, high)
        ch   = (ch - low) / (high - low + 1e-6)
        out[:, :, c] = ch * 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# Elastic deformation transform (applied to numpy arrays)
if ALBUMENTATIONS:
    ELASTIC = A.ElasticTransform(
        alpha=120,
        sigma=120 * 0.05,
        p=0.5
    )


class GlaSDataset(Dataset):
    def __init__(self, img_dir, img_size=128, augment=False, split="train"):
        self.img_dir  = Path(img_dir)
        self.img_size = img_size
        self.augment  = augment
        self.split    = split

        self.images = sorted([
            f for f in self.img_dir.glob("*.bmp")
            if "anno" not in f.name
        ])

        print(f"GlaS {split}: {len(self.images)} images")

        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return len(self.images)

    def _get_anno_path(self, img_path):
        return img_path.parent / f"{img_path.stem}_anno.bmp"

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        anno_path = self._get_anno_path(img_path)

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(anno_path).convert("L")

        # Stain normalization
        image = stain_normalize(image)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # 90-degree rotations
            if random.random() > 0.5:
                angle = 90 * random.choice([1, 2, 3])
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)

            # Mild affine rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)

            # Elastic deformation
            if ALBUMENTATIONS and random.random() > 0.5:
                img_np  = np.array(image)
                mask_np = np.array(mask)
                result  = ELASTIC(image=img_np, mask=mask_np)
                image   = Image.fromarray(result["image"])
                mask    = Image.fromarray(result["mask"])

            # Color jitter
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.15, contrast=0.15,
                    saturation=0.15, hue=0.03
                )(image)

        img_t  = self.normalize(transforms.ToTensor()(image))
        mask_t = (transforms.ToTensor()(mask) > 0).float()

        return img_t, mask_t


def get_loaders(img_size=128, batch_size=4, num_workers=4):
    train_ds = GlaSDataset(TRAIN_DIR, img_size=img_size, augment=True,  split="train")
    test_ds  = GlaSDataset(TEST_DIR,  img_size=img_size, augment=False, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    imgs, masks = next(iter(train_loader))
    print(f"Train: {imgs.shape}, {masks.shape}")
    imgs, masks = next(iter(test_loader))
    print(f"Test:  {imgs.shape}, {masks.shape}")