from pathlib import Path
import re
import numpy as np

from monai.transforms import (
    Compose,                    #Führe mehrere Transform-Schritte nacheinander aus.
    LoadImaged,                 #LoadImaged(keys=["image", "label"]) == lade die Datei unter "image" und die Datei unter "label" 
    EnsureChannelFirstd,        #Wandelt das Eingabeformat von (D, H, W) in (1, D, H, W) um, damit das 3D-Bild direkt in einen PyTorch-Netzwerkinput passt
    ScaleIntensityRanged,       #Das normalisiert die CT-Werte.
    Lambdad,                    #Eigenen kleinen Python-Schritt in die Pipeline einbauen --> Hier benutzen wir es, um die Maske binär zu machen
    RandCropByPosNegLabeld,     #Crop: schneidet manchmal positiv um Foreground herum; manchmal negativ aus Hintergrundregionen
    EnsureTyped,                #Das macht daraus am Ende saubere MONAI-/PyTorch-Tensoren.
)
from monai.data import CacheDataset, ITKReader


def get_lndb_datalist(
    image_dir,
    mask_dir,
    reader="rad2",
    max_cases=None,
):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted(image_dir.glob("LNDb-*.mhd"))
    mask_paths = sorted(mask_dir.glob(f"LNDb-*_{reader}.mhd"))

    image_map = {p.stem: str(p) for p in image_paths}

    mask_map = {}
    for p in mask_paths:
        match = re.match(r"(LNDb-\d+)_rad\d+", p.stem)
        if match:
            case_id = match.group(1)
            mask_map[case_id] = str(p)

    common_ids = sorted(set(image_map.keys()) & set(mask_map.keys()))
    if max_cases is not None:
        common_ids = common_ids[:max_cases]

    datalist = [
        {"image": image_map[case_id], "label": mask_map[case_id], "case_id": case_id}
        for case_id in common_ids
    ]
    return datalist


def get_train_transforms(patch_size=(64, 64, 64), num_samples=2):
    reader = ITKReader()
    return Compose([
        LoadImaged(keys=["image", "label"], reader=reader),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,        #Alles unter -1000 HU abschneiden
            a_max=400,          #Alles über 400 abschneiden
            b_min=0.0,          #Dann auf 0 bis 1 skalieren
            b_max=1.0,
            clip=True,
        ),
        Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_train_dataset(
    image_dir,
    mask_dir,
    reader="rad2",
    patch_size=(64, 64, 64),
    num_samples=2,
    max_cases=None,
    cache_rate=1.0,
):
    datalist = get_lndb_datalist(
        image_dir=image_dir,
        mask_dir=mask_dir,
        reader=reader,
        max_cases=max_cases,
    )

    transforms = get_train_transforms(
        patch_size=patch_size,
        num_samples=num_samples,
    )

    dataset = CacheDataset(
        data=datalist,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=0,
    )
    return dataset
