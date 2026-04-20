import matplotlib.pyplot as plt
import numpy as np
from data_loader3D import Luna25Dataset

CSV_PATH = '../data/test_subset/metadata/LUNA25_Public_Training_Development_Data_with_metadata.csv'
IMG_DIR = '../data/test_subset/images/'

def test_visualization():
    dataset = Luna25Dataset(CSV_PATH, IMG_DIR)
    print(f"Dataset geladen. Anzahl Bilder: {len(dataset)}")

    image, mask = dataset[234]
    
    mask_np = mask.squeeze().numpy()
    z_indices = np.where(mask_np > 0)[0]
    
    if len(z_indices) > 0:
        slice_idx = int(np.median(z_indices))
        print(f"Nodule gefunden auf Slice: {slice_idx}")
    else:
        slice_idx = 32 # Fallback
        print("WARNUNG: Keine Maske im gesamten Volumen gefunden!")

    img_slice = image[0, slice_idx, :, :].numpy()
    mask_slice = mask_np[slice_idx, :, :]

    mask_display = mask_slice.copy()
    mask_display[mask_display == 0] = np.nan

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title(f"CT-Bild (Slice {slice_idx})")

    plt.subplot(1, 2, 2)
    plt.imshow(img_slice, cmap='gray')
    # Overlay nur zeichnen, wenn Werte vorhanden sind
    plt.imshow(mask_display, cmap='Reds', alpha=0.6, vmin=0, vmax=1) 
    plt.title("Kugel-Maske Overlay")

    plt.show()

if __name__ == "__main__":
    test_visualization()