import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model3D import unet


# from utils import (load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160      # change
IMAGE_WIDTH = 240       # change
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
  loop = tqdm(loader)

  for batch_index, (data, targets) in enumerate(loop):
    data = data.to(device = DEVICE)
    targets = targets.float().unsqueeze(1).to(device=DEVICE)

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
  train.transform = A.Compose([A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
                               A.Rotate(limit = 35, p = 1.0),
                               A.HorizontalFlip(p = 0.5),
                               A.VerticalFlip(p = 0.1),
                               A.Normalize(mean = [0.0, 0.0, 0.0], std = [1.0, 1.0, 1.0], max_pixel_value = 255.0), ToTensorV2()])
  

  val_transforms = A.Compose([A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
                              A.Normalize(mean = [0.0, 0.0, 0.0], std = [1.0, 1.0, 1.0], max_pixel_value = 255.0), ToTensorV2()])
  
  model = unet(in_channels = 3, out_channels = 1).to(DEVICE)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)


  train_loader, vall_loader = get_loaders()

if __name__ == "__main__":
  main()