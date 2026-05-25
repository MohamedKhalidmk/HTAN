import os
import shutil
import zipfile
import random
import requests
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = Path("/opt/dlami/nvme/HTAN/data/isic")
TRAIN_IMG_DIR  = BASE_DIR / "train_images"
TRAIN_MASK_DIR = BASE_DIR / "train_masks"

VAL_SIZE = 520   # fixed — last 520 of shuffled list become validation

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
URLS = {
    "train_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
    "train_masks":  "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
}

def _download_and_extract(url: str, zip_name: str, final_dest: Path):
    tmp_zip = BASE_DIR / zip_name
    tmp_dir = BASE_DIR / (zip_name.replace(".zip", "_tmp"))

    print(f"  Downloading {zip_name}...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for {url}")

    with open(tmp_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"  Extracting {zip_name}...")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(tmp_dir)

    tmp_zip.unlink()

    # ISIC zips contain one subfolder — move its contents to final_dest
    final_dest.mkdir(parents=True, exist_ok=True)
    subfolder = next(tmp_dir.iterdir())
    files = list(subfolder.iterdir())
    for f in files:
        dst = final_dest / f.name
        if dst.exists():
            dst.unlink()
        shutil.move(str(f), str(dst))

    shutil.rmtree(tmp_dir)
    print(f"  Done: {final_dest} ({len(files)} files)")


def download_isic():
    """Download ISIC-2018 training images and masks to data/isic/."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_IMG_DIR.exists() and len(list(TRAIN_IMG_DIR.glob("*.jpg"))) > 2000:
        print("ISIC images already downloaded, skipping.")
    else:
        print("Downloading ISIC-2018 training images...")
        _download_and_extract(URLS["train_images"], "train_img.zip", TRAIN_IMG_DIR)

    if TRAIN_MASK_DIR.exists() and len(list(TRAIN_MASK_DIR.glob("*.png"))) > 2000:
        print("ISIC masks already downloaded, skipping.")
    else:
        print("Downloading ISIC-2018 training masks...")
        _download_and_extract(URLS["train_masks"], "train_msk.zip", TRAIN_MASK_DIR)

    print("\nISIC-2018 ready.")


# ---------------------------------------------------------------------------
# Internal subset dataset
# ---------------------------------------------------------------------------
class _ISICSubset(Dataset):
    """Takes a pre-filtered list of image names — no double directory scan."""

    def __init__(self, img_dir, mask_dir, image_names, img_size=256, augment=False):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.images    = image_names
        self.img_size  = img_size
        self.augment   = augment
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.rsplit(".", 1)[0] + "_segmentation.png"

        image = Image.open(self.img_dir / img_name).convert("RGB")
        mask  = Image.open(self.mask_dir / mask_name).convert("L")

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size),  Image.Resampling.NEAREST)

        # Augmentation — paper faithful
        if self.augment:
            if random.random() > 0.5:
                angle = random.uniform(-20, 20)
                scale = random.uniform(0.9, 1.1)
                image = TF.affine(image, angle=angle, translate=(0, 0), scale=scale, shear=0,
                                  interpolation=TF.InterpolationMode.BILINEAR)
                mask  = TF.affine(mask,  angle=angle, translate=(0, 0), scale=scale, shear=0,
                                  interpolation=TF.InterpolationMode.NEAREST)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1
                )(image)

        img_t  = self.normalize(transforms.ToTensor()(image))
        mask_t = (transforms.ToTensor()(mask) > 0.5).float()

        return img_t, mask_t


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def get_loaders(img_size: int = 256, batch_size: int = 4,
                seed: int = 42, num_workers: int = 4):
    """
    Returns (train_loader, val_loader).

    Split: images sorted alphabetically, shuffled with seed,
    last VAL_SIZE become validation, rest become training.
    """
    all_images = sorted([
        f.name for f in TRAIN_IMG_DIR.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    total = len(all_images)

    g = torch.Generator()
    g.manual_seed(seed)
    indices      = torch.randperm(total, generator=g).tolist()
    train_images = [all_images[i] for i in indices[:total - VAL_SIZE]]
    val_images   = [all_images[i] for i in indices[total - VAL_SIZE:]]

    print(f"ISIC — Total: {total} | Train: {len(train_images)} | Val: {len(val_images)}")

    train_ds = _ISICSubset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_images,
                           img_size=img_size, augment=True)
    val_ds   = _ISICSubset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, val_images,
                           img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True       # avoids unstable BatchNorm on small last batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader = get_loaders()
    imgs, masks = next(iter(train_loader))
    print(f"Train batch — images: {imgs.shape}, masks: {masks.shape}")
    imgs, masks = next(iter(val_loader))
    print(f"Val batch   — images: {imgs.shape}, masks: {masks.shape}")