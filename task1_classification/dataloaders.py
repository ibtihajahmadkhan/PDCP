import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST
import os

def get_pneumoniamnist_loaders(
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    img_size: int = 224,
):
    """
    Train: augmentation (±7° rotation, small translation)
    Val/Test: deterministic preprocessing only
    ResNet18 works better when upsampling from 28x28 -> 224x224.
    """

    os.makedirs(data_root, exist_ok=True)
    # Train augmentation: small rotations and translations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=7,
            translate=(0.05, 0.05),  # 5% shift
            fill=0
        ),
        transforms.ToTensor(),
        # ImageNet-like normalization but single-channel (since we changed conv1 to 1-ch)
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    # Eval transform: no augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    train_ds = PneumoniaMNIST(split="train", root=data_root, transform=train_transform, download=True)
    val_ds   = PneumoniaMNIST(split="val",   root=data_root, transform=eval_transform,  download=True)
    test_ds  = PneumoniaMNIST(split="test",  root=data_root, transform=eval_transform,  download=True)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader
