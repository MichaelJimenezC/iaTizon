import argparse, torch, numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .model import ResNet18
from .data import MEAN, STD

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)    # C:\dev\dataset_bin
    ap.add_argument("--ckpt", required=True)        # stage1_maize_not.pth
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--device", default="cuda")
    args=ap.parse_args()

    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    tf=transforms.Compose([
        transforms.Resize(int(args.img_size*1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])

    val_ds=datasets.ImageFolder(f"{args.data_dir}\\valid", transform=tf)
    val_ld=DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    classes=val_ds.classes                    # ['maize','not_maize'] (verifica)
    maize_idx=classes.index("maize")

    model=ResNet18(num_classes=2).to(device).eval()
    state=torch.load(args.ckpt, map_location=device); state=state.get("model", state)
    model.load_state_dict(state, strict=False)

    y_true=[]; p_maize=[]
    with torch.no_grad():
        for x,y in val_ld:
            x=x.to(device)
            p=torch.softmax(model(x),1)[:,maize_idx].cpu().numpy()
            p_maize.extend(p)
            y_true.extend((y.numpy()==maize_idx).astype(int))  # 1 si maize, 0 si not_maize

    y_true=np.array(y_true); p_maize=np.array(p_maize)

    prec, rec, thr = precision_recall_curve(y_true, p_maize)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    best = int(np.nanargmax(f1))
    tau  = float(thr[max(0,best-1)])  # umbral asociado al mejor F1
    print(f"tau ≈ {tau:.3f}")
    open("stage1_tau.txt","w").write(str(tau))

if __name__=="__main__":
    main()
