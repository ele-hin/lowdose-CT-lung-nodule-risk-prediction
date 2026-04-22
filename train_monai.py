import torch
from monai.data import DataLoader, list_data_collate

from dataset_monai import get_train_dataset
from UNET.unet import UNet


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_dir = "/Users/luis/Desktop/UNI/Praktika/BIH_Lungnodules/data/lndb/extracted/data0/data0"
    mask_dir = "/Users/luis/Desktop/UNI/Praktika/BIH_Lungnodules/data/lndb/extracted/masks/masks"

    dataset = get_train_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        reader="rad2",
        patch_size=(64, 64, 64),
        num_samples=2,
        max_cases=10,
        cache_rate=1.0,
    )

    print("Number of cases:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
    )

    model = UNet(in_channels=1, num_classes=1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for step, batch in enumerate(dataloader):
            images = batch["image"].float().to(device)
            masks = batch["label"].float().to(device)

            if epoch == 0 and step == 0:
                print("First batch image shape:", images.shape)
                print("First batch mask shape:", masks.shape)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    train()
