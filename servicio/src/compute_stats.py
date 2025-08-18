import os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    data_dir = r"C:\dev\dataset"
    img_size = 320
    batch_size = 64
    tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    n = 0
    s = torch.zeros(3)
    ss = torch.zeros(3)
    for x,_ in dl:
        b = x.size(0)*x.size(2)*x.size(3)
        s += x.sum(dim=[0,2,3])
        ss += (x**2).sum(dim=[0,2,3])
        n += b
    mean = (s/n).tolist()
    std = torch.sqrt(ss/n - (s/n)**2).tolist()
    print("MEAN =", tuple(float(f"{m:.6f}") for m in mean))
    print("STD  =", tuple(float(f"{v:.6f}") for v in std))

if __name__ == "__main__":
    main()
