import math

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from config import (
    BACKGROUND_AS_CLASS,
    BCE_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    TRAIN_CUDA,
    TRAINING_EPOCH,
)
from dataset import get_train_val_test_data_loaders
from transforms import (
    train_transforms,
    train_transforms_cuda,
    val_transforms,
    val_transforms_cuda,
)
from unet3d import UNet3D

if BACKGROUND_AS_CLASS:
    NUM_CLASSES += 1


writer = SummaryWriter("runs")


model = UNet3D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
train_transforms = train_transforms
val_transforms = val_transforms

if torch.cuda.is_available() and TRAIN_CUDA:
    model = model.cuda()
    train_transforms = train_transforms_cuda
    val_transforms = val_transforms_cuda
elif not torch.cuda.is_available() and TRAIN_CUDA:
    print("cuda not available! Training initialized on cpu ...")


train_dataloader, val_dataloader, _ = get_train_val_test_data_loaders(
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    test_transforms=val_transforms,
)

criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
optimizer = Adam(model.parameters(), lr=1e-4)

min_valid_loss = math.inf

for epoch in range(TRAINING_EPOCH):

    train_loss = 0.0
    model.train()

    for data in train_dataloader:
        image, ground_truth = data["image"], data["label"]
        target = model(image)
        loss = criterion(target, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()

    for data in val_dataloader:
        image, ground_truth = data["image"], data["label"]

        target = model(image)
        loss = criterion(target, ground_truth)
        valid_loss = loss.item()

    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)

    print(
        f"Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}"
    )

    if min_valid_loss > valid_loss:
        print(
            f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
        )
        min_valid_loss = valid_loss
        # Saving state dict
        torch.save(
            model.state_dict(), f"checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth"
        )

writer.flush()
writer.close()
