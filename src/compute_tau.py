import argparse, torch, numpy as np
from sklearn.metrics import precision_recall_curve
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .model import ResNet18
from .data import MEAN, STD

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pos_class", type=str, default="maize",
                    help="Clase positiva para calcular tau (p.ej. 'maize' o 'tizon_foliar')")
    args=ap.parse_args()

    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    tf=transforms.Compose([
        transforms.Resize(int(args.img_size*1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])

    val_ds=datasets.ImageFolder(f"{args.data_dir}\\valid", transform=tf)
    classes=val_ds.classes
    if args.pos_class not in classes:
        raise ValueError(f"'{args.pos_class}' no está en las clases {classes}")
    pos_idx=classes.index(args.pos_class)

    val_ld=DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model=ResNet18(num_classes=len(classes)).to(device).eval()
    state=torch.load(args.ckpt, map_location=device); state=state.get("model", state)
    model.load_state_dict(state, strict=False)

    y_true=[]; p_pos=[]
    with torch.no_grad():
        for x,y in val_ld:
            x=x.to(device)
            p=F.softmax(model(x),1)[:,pos_idx].cpu().numpy()
            p_pos.extend(p)
            y_true.extend((y.numpy()==pos_idx).astype(int))

    y_true=np.array(y_true); p_pos=np.array(p_pos)
    prec, rec, thr = precision_recall_curve(y_true, p_pos)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    best = int(np.nanargmax(f1))
    tau  = float(thr[max(0,best-1)])
    print(f"tau ≈ {tau:.3f}")
    open("tau.txt","w").write(str(tau))

if __name__=="__main__":
    main()
