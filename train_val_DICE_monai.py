import random
import torch
import SimpleITK as sitk
import numpy as np

from monai.data import DataLoader, list_data_collate, CacheDataset
from monai.losses import DiceCELoss

from dataset_monai import (
    get_lndb_datalist,
    get_train_transforms,
    get_val_transforms,
)
from UNET.unet import UNet


def filter_empty_masks(datalist):
    filtered = []
    for item in datalist:
        mask_img = sitk.ReadImage(item["label"])
        mask_arr = sitk.GetArrayFromImage(mask_img)
        if np.any(mask_arr > 0):
            filtered.append(item)
    return filtered


def split_datalist(datalist, train_ratio=0.7, val_ratio=0.15, seed=42):
    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1.0"

    datalist = datalist.copy()
    rng = random.Random(seed)
    rng.shuffle(datalist)

    n = len(datalist)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_files = datalist[:train_end]
    val_files = datalist[train_end:val_end]
    test_files = datalist[val_end:]

    return train_files, val_files, test_files


def build_datasets(image_dir, mask_dir, max_cases=20, seed=42):
    # 1) Alle CT-Masken-Paare holen
    datalist = get_lndb_datalist(
        image_dir=image_dir,
        mask_dir=mask_dir,
        reader="rad2",
        max_cases=max_cases,
    )

    print("Total selected cases before filtering:", len(datalist))

    # 2) Leere Masken entfernen
    datalist = filter_empty_masks(datalist)
    print("Cases after removing empty masks:", len(datalist))

    # 3) In train / val / test splitten
    train_files, val_files, test_files = split_datalist(
        datalist,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=seed,
    )

    print("Train cases:", len(train_files))
    print("Val cases:", len(val_files))
    print("Test cases:", len(test_files))

    # 4) Transforms
    train_transforms = get_train_transforms(
        patch_size=(64, 64, 64),
        num_samples=2,
    )

    val_transforms = get_val_transforms(
        patch_size=(64, 64, 64),
    )

    # 5) Datasets
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=0,
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=0,
    )

    test_ds = CacheDataset(
        data=test_files,
        transform=val_transforms,   # test soll auch deterministisch sein
        cache_rate=1.0,
        num_workers=0,
    )

    return train_ds, val_ds, test_ds


def build_loaders(train_ds, val_ds, test_ds):
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader, test_loader


def train_and_validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_dir = "/Users/luis/Desktop/UNI/Praktika/BIH_Lungnodules/data/lndb/extracted/data0/data0"
    mask_dir = "/Users/luis/Desktop/UNI/Praktika/BIH_Lungnodules/data/lndb/extracted/masks/masks"

    train_ds, val_ds, test_ds = build_datasets(
        image_dir=image_dir,
        mask_dir=mask_dir,
        max_cases=20,
        seed=42,
    )

    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds)

    print("Train loader batches:", len(train_loader))
    print("Val loader batches:", len(val_loader))
    print("Test loader batches:", len(test_loader))

    model = UNet(in_channels=1, num_classes=1).to(device)
    criterion = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ===== TRAIN =====
        model.train()
        train_running_loss = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].float().to(device)
            masks = batch["label"].float().to(device)

            if epoch == 0 and step == 0:
                print("First TRAIN batch image shape:", images.shape)
                print("First TRAIN batch mask shape:", masks.shape)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

        train_loss = train_running_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                images = batch["image"].float().to(device)
                masks = batch["label"].float().to(device)

                if epoch == 0 and step == 0:
                    print("First VAL batch image shape:", images.shape)
                    print("First VAL batch mask shape:", masks.shape)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

        # bestes Modell speichern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_monai_lndb_dicece.pth")
            print("Saved new best model.")

    # letztes Modell optional auch speichern
    torch.save(model.state_dict(), "last_unet_monai_lndb_dicece.pth")
    print("Saved final model.")


def evaluate_test(model_path="best_unet_monai_lndb.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_dir = "/Users/luis/Desktop/UNI/Praktika/BIH_Lungnodules/data/lndb/extracted/data0/data0"
    mask_dir = "/Users/luis/Desktop/UNI/Praktika/BIH_Lungnodules/data/lndb/extracted/masks/masks"

    train_ds, val_ds, test_ds = build_datasets(
        image_dir=image_dir,
        mask_dir=mask_dir,
        max_cases=20,
        seed=42,
    )

    _, _, test_loader = build_loaders(train_ds, val_ds, test_ds)

    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
    test_running_loss = 0.0

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            images = batch["image"].float().to(device)
            masks = batch["label"].float().to(device)

            if step == 0:
                print("First TEST batch image shape:", images.shape)
                print("First TEST batch mask shape:", masks.shape)

            outputs = model(images)
            loss = criterion(outputs, masks)
            test_running_loss += loss.item()

    test_loss = test_running_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    train_and_validate()
    # evaluate_test()   # <-- erstmal AUSKOMMENTIERT lassen