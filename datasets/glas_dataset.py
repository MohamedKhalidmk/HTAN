"""
glas_dataset.py — GlaS Dataset Loader
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


DATA_ROOT = Path("/opt/dlami/nvme/HTAN/data/glas")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"


def download_glas():
    if TRAIN_DIR.exists() and len(list(TRAIN_DIR.glob("*.bmp"))) >= 170:
        print("GlaS data already exists.")
        return

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print("GlaS dataset requires manual download.")
    print("Download from:")
    print("  https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download/")
    print(f"Extract to: {DATA_ROOT}")


def simple_stain_normalize_pil(image):
    """
    Lightweight stain/color normalization for H&E-style histology.
    This is not Macenko. It is intentionally simple and safe:
    per-channel percentile clipping + contrast normalization.
    """
    arr = np.asarray(image).astype(np.float32)

    out = np.zeros_like(arr)

    for c in range(3):
        channel = arr[:, :, c]
        low = np.percentile(channel, 1)
        high = np.percentile(channel, 99)

        channel = np.clip(channel, low, high)
        channel = (channel - low) / (high - low + 1e-6)
        out[:, :, c] = channel * 255.0

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


class GlaSDataset(Dataset):
    def __init__(self, img_dir, img_size=128, augment=False, split="train"):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment
        self.split = split

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
        name = img_path.stem
        return img_path.parent / f"{name}_anno.bmp"

    def __getitem__(self, idx):
        img_path = self.images[idx]
        anno_path = self._get_anno_path(img_path)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(anno_path).convert("L")

        # Stain/color normalization before resize
        image = simple_stain_normalize_pil(image)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            # Flips
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 90-degree rotations are safe for gland morphology
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                angle = 90 * k
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

            # Mild rotation only, no heavy shear/translation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=Image.BILINEAR
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=Image.NEAREST
                )

            # Mild color jitter after stain normalization
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.10,
                    hue=0.02
                )(image)

        img_t = self.normalize(transforms.ToTensor()(image))
        mask_t = (transforms.ToTensor()(mask) > 0).float()

        return img_t, mask_t


def get_loaders(img_size=128, batch_size=4, num_workers=4):
    train_ds = GlaSDataset(
        TRAIN_DIR,
        img_size=img_size,
        augment=True,
        split="train"
    )

    test_ds = GlaSDataset(
        TEST_DIR,
        img_size=img_size,
        augment=False,
        split="test"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    download_glas()
    train_loader, test_loader = get_loaders()
    imgs, masks = next(iter(train_loader))
    print(f"Train batch — images: {imgs.shape}, masks: {masks.shape}")
    imgs, masks = next(iter(test_loader))
    print(f"Test batch  — images: {imgs.shape}, masks: {masks.shape}")