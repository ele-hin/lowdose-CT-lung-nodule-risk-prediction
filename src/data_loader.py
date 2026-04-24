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
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        # Euklidscher Abstand
        distance = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
        return (distance <= radius**2).astype(np.float32)
        

class Luna25Dataset(Dataset):
    """
    3D-Dataset-Loader für Luna25

    :param Dataset:
    :return:
    """
    def __init__(self, csv_path, image_dir, mask_dir = None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # Filter, dass nur Dateien, die auch in CSV sind, geladen werden
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]

        if self.mask_dir:
            os.makedirs(self.mask_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
         img_name = self.image_files[idx]
         annotation_id = img_name.replace('.npy', '')
         #row = self.df[self.df['AnnotationID'] == annotation_id].iloc[0]

         image = np.load(os.path.join(self.image_dir, img_name)).astype(np.float32)

         image = np.clip(image, -1000, 400)
         image = (image + 1000) / 1400

         #vox_z = (row['CoordZ'] - row['z_origin']) / row['z_spacing']
         #vox_y = (row['CoordY'] - row['y_origin']) / row['y_spacing']
         #vox_x = (row['CoordX'] - row['x_origin']) / row['x_spacing']

         mask = create_sphere_mask(image.shape, (32, 64, 64), radius=8)

         if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, img_name)
            if not os.path.exists(mask_path):
                np.save(mask_path, mask)

         image_tensor = torch.from_numpy(image).unsqueeze(0)
         mask_tensor = torch.from_numpy(mask).unsqueeze(0)
         #coords = {'z': vox_z, 'y': vox_y, 'x': vox_x}

         #print(f"ID: {annotation_id} | Voxel-Pos: Z={vox_z:.1f}, Y={vox_y:.1f}, X={vox_x:.1f}")

         return image_tensor, mask_tensor, annotation_id


# import torch
# from torch.utils.data import Dataset
# import os
# import numpy as np

# class Luna25ProcessedDataset(Dataset):
#     def __init__(self, processed_dir):
#         self.img_dir = os.path.join(processed_dir, "images")
#         self.mask_dir = os.path.join(processed_dir, "masks")
#         self.filenames = [f for f in os.listdir(self.img_dir) if f.endswith('.npy')]

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         name = self.filenames[idx]
#         image = np.load(os.path.join(self.img_dir, name))
#         mask = np.load(os.path.join(self.mask_dir, name))
        
#         # PyTorch erwartet (Channel, Z, Y, X)
#         return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0).float()





# import os
# import SimpleITK as sitk
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# def create_sphere_mask(shape, center, radius):
#         """
#         Erzeugt 3D-Kugelmaske mit 1 = Nodule, 0 = Hintergrund
#         shape: (Z, Y, X)
#         center: (v_z, v_y, v_x)
#         """
#         # erzeugt Koordinaten-Gitter
#         z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
#         # Euklidscher Abstand
#         distance = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
#         return (distance <= radius**2).astype(np.float32)
        

# class Luna25Dataset(Dataset):
#     """
#     3D-Dataset-Loader für Luna25

#     :param Dataset:
#     :return:
#     """
#     def __init__(self, csv_path, mha_dir, crop_size=(64, 128, 128)):
#         self.df = pd.read_csv(csv_path)
#         self.mha_dir = mha_dir
#         self.crop_size = crop_size
#         # Filter, dass nur Dateien, die auch in CSV sind, geladen werden
#         self.active_uids = [f.replace('.mha', '') for f in os.listdir(mha_dir) if f.endswith('.mha')]

#         self.samples = self.df[self.df['SeriesInstanceUID'].isin(self.active_uids)].copy()
#         print(f"Mapping finished: {len(self.active_uids)} scans loaded, {len(self.samples)} nodules found.")

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#          row = self.samples.iloc[idx]
#          uid = row['SeriesInstanceUID']
#          mha_path = os.path.join(self.mha_dir, f"{uid}.mha")

#          itk_img = sitk.ReadImage(mha_path)
#          img_array = sitk.GetArrayFromImage(itk_img).astype(np.float32)

#          world_coords = (row['CoordX'], row['CoordY'], row['CoordZ'])
#          voxel_center = itkg_img.TransformPhysicalPointToIndex(world_coords)
#          v_z, v_y, v_x = voxel_center[0], voxel_center[1], voxel_center[2]
#          #annotation_id = img_name.replace('.npy', '')
#          #row = self.df[self.df['AnnotationID'] == annotation_id].iloc[0]

#          image = np.load(os.path.join(self.image_dir, img_name)).astype(np.float32)

#          image = np.clip(image, -1000, 400)
#          image = (image + 1000) / 1400

#          #vox_z = (row['CoordZ'] - row['z_origin']) / row['z_spacing']
#          #vox_y = (row['CoordY'] - row['y_origin']) / row['y_spacing']
#          #vox_x = (row['CoordX'] - row['x_origin']) / row['x_spacing']

#          mask = create_sphere_mask(image.shape, (32, 64, 64), radius=8)

#          image = torch.from_numpy(image).unsqueeze(0)
#          mask = torch.from_numpy(mask).unsqueeze(0)
#          #coords = {'z': vox_z, 'y': vox_y, 'x': vox_x}

#          #print(f"ID: {annotation_id} | Voxel-Pos: Z={vox_z:.1f}, Y={vox_y:.1f}, X={vox_x:.1f}")

#          return image, mask



# def load_mha_image(file_path):
#     image = sitk.ReadImage(file_path)
#     array = sitk.GetArrayFromImage(image)
#     return array

# def load_npy_image(file_path):
#     array = np.load(file_path, allow_pickle=True)
#     return array

# class Luna25Dataset(Dataset):
#     """
#     3D-Dataset-Loader für Luna25

#     :param Dataset:
#     :return:
#     """
#     def __init__(self, image_dir, mask_dir):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.image_files = sorted(os.listdir(image_dir))

#     def __len__(self):
#         return len(self.image_files)
    
#     def __getitem__(self, index):
#         img_name = self.image_files[index]
#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, img_name)

#         image_array = load_npy_image(img_path)
#         mask_array = load_npy_image(mask_path)

#         image_array = image_array.astype(np.float32)
#         mask_array = mask_array.astype(np.float32)

#         #TODO: Normalisierung der images noch besprechen später

#         image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
#         mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)

#         return image_tensor, mask_tensor
