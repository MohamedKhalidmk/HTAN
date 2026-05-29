"""
bowl_dataset.py — 2018 Data Science Bowl Dataset Loader

Dataset: 2018 Data Science Bowl (nuclei segmentation)
Kaggle: https://www.kaggle.com/c/data-science-bowl-2018

Download on AWS:
    kaggle competitions download -c data-science-bowl-2018 -p /opt/dlami/nvme/HTAN/data/bowl/
    cd /opt/dlami/nvme/HTAN/data/bowl/
    unzip data-science-bowl-2018.zip

Structure after download:
    stage1_train/
        <image_id>/
            images/
                <image_id>.png
            masks/
                <mask_id>.png   (one file per nucleus)
    stage1_test/

Split: 80/10/10 from 671 training images (seed 123)
Image size: 256x256 (following paper)
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


DATA_ROOT  = Path("/opt/dlami/nvme/HTAN/data/bowl")
TRAIN_ROOT = DATA_ROOT / "stage1_train"


def get_split(seed=123):
    """
    Split 671 training images into 80/10/10 following paper.
    Returns (train_ids, val_ids, test_ids)
    """
    all_ids = sorted([d.name for d in TRAIN_ROOT.iterdir() if d.is_dir()])
    assert len(all_ids) > 0, f"No images found in {TRAIN_ROOT}"

    # 80% train, 20% temp
    train_ids, temp_ids = train_test_split(
        all_ids, test_size=0.2, random_state=seed
    )
    # 50/50 split of temp → 10% val, 10% test
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=seed
    )

    return train_ids, val_ids, test_ids


def merge_masks(mask_dir):
    """Merge multiple per-nucleus binary masks into one binary mask."""
    mask_files = list(Path(mask_dir).glob("*.png"))
    if not mask_files:
        return None

    # Read first mask to get size
    first = np.array(Image.open(mask_files[0]).convert("L"))
    merged = np.zeros_like(first, dtype=np.uint8)

    for mf in mask_files:
        m = np.array(Image.open(mf).convert("L"))
        merged = np.maximum(merged, m)

    return Image.fromarray((merged > 0).astype(np.uint8) * 255)


class BowlDataset(Dataset):
    def __init__(self, image_ids, img_size=256, augment=False, split="train"):
        self.image_ids = image_ids
        self.img_size  = img_size
        self.augment   = augment
        self.split     = split

        print(f"Bowl {split}: {len(self.image_ids)} images")

        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id    = self.image_ids[idx]
        img_path  = TRAIN_ROOT / img_id / "images" / f"{img_id}.png"
        mask_dir  = TRAIN_ROOT / img_id / "masks"

        image = Image.open(img_path).convert("RGB")
        mask  = merge_masks(mask_dir)

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

            # Mild affine
            if random.random() > 0.5:
                angle = random.uniform(-20, 20)
                scale = random.uniform(0.9, 1.1)
                image = TF.affine(image, angle=angle, translate=(0, 0),
                                  scale=scale, shear=0,
                                  interpolation=Image.BILINEAR)
                mask  = TF.affine(mask, angle=angle, translate=(0, 0),
                                  scale=scale, shear=0,
                                  interpolation=Image.NEAREST)

            # Color jitter
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.05
                )(image)

        img_t  = self.normalize(transforms.ToTensor()(image))
        mask_t = (transforms.ToTensor()(mask) > 0.5).float()

        return img_t, mask_t


def get_loaders(img_size=256, batch_size=4, seed=123, num_workers=4):
    """
    Returns (train_loader, val_loader, test_loader)
    Split: 80/10/10 from 671 training images
    """
    train_ids, val_ids, test_ids = get_split(seed=seed)

    train_ds = BowlDataset(train_ids, img_size=img_size, augment=True,  split="train")
    val_ds   = BowlDataset(val_ids,   img_size=img_size, augment=False, split="val")
    test_ds  = BowlDataset(test_ids,  img_size=img_size, augment=False, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    imgs, masks = next(iter(train_loader))
    print(f"Train: {imgs.shape}, {masks.shape}")
    imgs, masks = next(iter(val_loader))
    print(f"Val:   {imgs.shape}, {masks.shape}")