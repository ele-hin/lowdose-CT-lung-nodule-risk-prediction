import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model2D import UNet2D
from data_loader2D import Luna25Dataset
from torch.utils.data import DataLoader


# from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 #moderate batchsize für Laptop
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 128      
IMAGE_WIDTH = 128       
PIN_MEMORY = True

CSV_PATH = '../data/test_subset/metadata/LUNA25_Public_Training_Development_Data_with_metadata.csv'
IMG_DIR = '../data/test_subset/images/'

def train_fn(loader, model, optimizer, loss_fn, scaler):
  loop = tqdm(loader)

  for batch_index, (data, targets) in enumerate(loop):
    data = data.to(device = DEVICE)

    if len(targets.shape) == 3:
      targets = targets.unsqueeze(1)

    targets = targets.float().to(device=DEVICE)

    # forward
    with torch.cuda.amp.autocast():
      predictions = model(data)
      loss = loss_fn(predictions, targets)

    # backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # update tqdm loop
    loop.set_postfix(loss = loss.item())

def main():
  train_transforms = A.Compose([
                               A.Rotate(limit = 35, p = 1.0),
                               A.HorizontalFlip(p = 0.5),
                               A.VerticalFlip(p = 0.1),
                               ToTensorV2()])
  
  #Modell Initialisierung
  model = UNet2D(in_channels = 1, out_channels = 1).to(DEVICE)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
  scaler = torch.cuda.amp.GradScaler()

  dataset = Luna25Dataset(csv_path= CSV_PATH, image_dir=IMG_DIR, transform = train_transforms)
  train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory = PIN_MEMORY,
    shuffle = True,
  )

  print(f"Starte Training auf {DEVICE}...")
  for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_fn(train_loader, model, optimizer, loss_fn, scaler)


if __name__ == "__main__":
  main()