import torch
from src.data import build_loaders

def main():
    data_dir    = r"C:\dev\dataset"
    img_size    = 320
    batch_size  = 64
    num_workers = 4
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers
    )
    print("Clases detectadas:", train_ds.classes)
    xb, yb = next(iter(train_loader))
    print("Batch shape:", xb.shape)
    print("Labels shape:", yb.shape)
    print("Ejemplo labels:", yb[:16].tolist())
    print("xb mean:", xb.mean().item(), "| std:", xb.std().item())

if __name__ == "__main__":
    main()
