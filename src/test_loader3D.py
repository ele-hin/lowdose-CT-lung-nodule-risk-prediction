import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os
import pandas as pd
from data_loader import Luna25Dataset

CSV_PATH = '../data/test_subset/metadata/LUNA25_Public_Training_Development_Data_with_metadata.csv'
IMG_DIR = '../data/test_subset/images/'
MHA_DIR = "../data/test_subset/luna25_images/"

def test_visualization():
    dataset = Luna25Dataset(CSV_PATH, IMG_DIR)
    print(f"Loading Dataset. Number of images: {len(dataset)}")
    df = pd.read_csv(CSV_PATH)

    for i in range(len(dataset)):
        image, mask, annotation_id = dataset[i]

        matching_rows = df[df['AnnotationID'] == annotation_id]
        if matching_rows.empty:
            continue
            
        node_info = matching_rows.iloc[0]
        uid = node_info['SeriesInstanceUID']

        potential_path = os.path.join(MHA_DIR, f"{uid}.mha")

        if os.path.exists(potential_path):
            mha_path = potential_path
            found_idx = i
            print(f"Index: {found_idx}, File: {os.path.basename(mha_path)}")
            break

    print(f"Load mha-file: {mha_path}...")
    itk_img = sitk.ReadImage(mha_path)
    full_img_array = sitk.GetArrayFromImage(itk_img) # Format: (Z, Y, X)
    
    # world_coords für SimpleITK müssen (X, Y, Z) sein
    world_coords = (float(node_info['CoordX']), float(node_info['CoordY']), float(node_info['CoordZ']))
    v_coords = itk_img.TransformPhysicalPointToIndex(world_coords)
    v_x, v_y, v_z = v_coords # SimpleITK gibt (x, y, z) zurück

    print(f"Nodule Welt-Koordinaten: {world_coords}")
    print(f"Nodule Voxel-Koordinaten: X={v_x}, Y={v_y}, Z={v_z}")


    mask_np = mask.squeeze().numpy()
    if mask_np.ndim == 3:
        mask_slice = mask_np[32, :, :] #32 noch debatierbar

    mask_display = mask_slice.copy()
    mask_display[mask_display == 0] = np.nan # Hintergrund transparent machen

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(full_img_array[v_z, :, :], cmap='gray')
    plt.scatter(v_x, v_y, s=100, edgecolors='r', facecolors='none')

    plt.show()

if __name__ == "__main__":
    test_visualization()
