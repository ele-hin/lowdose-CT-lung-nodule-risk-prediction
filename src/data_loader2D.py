import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def create_sphere_mask(shape, center, radius):
        """
        Erzeugt 3D-Kugelmaske mit 1 = Nodule, 0 = Hintergrund
        shape: (Z, Y, X)
        center: (v_z, v_y, v_x)
        """
        # erzeugt Koordinaten-Gitter
        y, x = np.ogrid[:shape[0], :shape[1]]
        # Euklidscher Abstand
        distance = (y - center[1])**2 + (x - center[2])**2
        return (distance <= radius**2).astype(np.float32)
        

class Luna25Dataset(Dataset):
    """
    3D-Dataset-Loader für Luna25

    :param Dataset:
    :return:
    """
    def __init__(self, csv_path, image_dir, transform = None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
         img_name = self.image_files[idx]

         image_3d = np.load(os.path.join(self.image_dir, img_name)).astype(np.float32)

         image_2d = image_3d[32, :, :]

         image_2d = np.clip(image_2d, -1000, 400)
         image_2d = (image_2d + 1000) / 1400

         mask_2d = create_sphere_mask(image_2d.shape, (64, 64, 64), radius=8)

         #Augmentierung
         if self.transform:
              augmentation = self.transform(image=image_2d, mask = mask_2d)
              image_2d = augmentation["image"]
              mask_2d = augmentation["mask"]
              
              return image_2d, mask_2d
