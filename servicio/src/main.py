import argparse, torch
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.backends.cudnn as cudnn
from .data import build_loaders
from .model import ResNet18
from .train import train_loop, evaluate

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"C:\dev\dataset")
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--step_size", type=int, default=40)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--ckpt", type=str, default="best_model_simple.pth")
    return ap.parse_args()

def main():
    args=parse_args()
    cudnn.benchmark=True
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds,val_ds,test_ds,_,val_loader,test_loader=build_loaders(
        data_dir=args.data_dir,img_size=args.img_size,batch_size=args.batch_size,num_workers=4
    )
    classes=train_ds.classes
    print("clases:", classes)

    cnt=Counter([y for _,y in train_ds.samples])
    weights=[1.0/cnt[y] for _,y in train_ds.samples]
    sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader=DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    model=ResNet18(num_classes=len(classes)).to(device)

    class_weights=None
    if not args.no_class_weights:
        total=sum(cnt.values()); C=len(classes)
        w=[total/(C*cnt[i]) for i in range(C)]
        class_weights=torch.tensor(w, dtype=torch.float)

    train_loop(model, train_loader, val_loader, device,
               epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
               step_size=args.step_size, gamma=args.gamma, amp=(not args.no_amp),
               class_weights=class_weights, ckpt_path=args.ckpt)

    ckpt=torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    crit=nn.CrossEntropyLoss(weight=(class_weights.to(device) if class_weights is not None else None))
    tl,ta,tf=evaluate(model, test_loader, device, crit, model.fc.out_features)
    print(f"TEST | loss {tl:.4f} | acc {ta:.4f} | f1 {tf:.4f}")

if __name__=="__main__":
    main()
