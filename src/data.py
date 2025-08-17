import os
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN = (0.462514, 0.538169, 0.393077)
STD  = (0.19825, 0.191944, 0.218688)


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15, fill=0),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

def build_loaders(
    data_dir: str,
    img_size: int = 320,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder,
           DataLoader, DataLoader, DataLoader]:
    train_tf = build_transforms(img_size, True)
    eval_tf  = build_transforms(img_size, False)
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
