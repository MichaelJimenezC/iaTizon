import argparse, torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt, numpy as np
from .data import build_loaders
from .model import ResNet18
from torchvision import transforms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"C:\dev\dataset")
    ap.add_argument("--ckpt", type=str, default="best_model_simple.pth")
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tau", type=float, default=None,
                    help="(Solo binario) Umbral para clase 'maize': si P(maize)<tau => 'not_maize'")
    ap.add_argument("--save_prefix", type=str, default="")  
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=4
    )
    classes = test_ds.classes  
    C = len(classes)

    model = ResNet18(num_classes=C).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    ys, yhat, p_maize = [], [], []
    maize_idx = classes.index("maize") if ("maize" in classes) else (None)

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, 1).cpu()
            if C == 2 and args.tau is not None and maize_idx is not None:
                p_m = probs[:, maize_idx].numpy()
                pred = np.where(p_m >= args.tau, maize_idx, 1 - maize_idx)
                yhat.append(torch.from_numpy(pred))
                p_maize.extend(p_m.tolist())
            else:
                yhat.append(probs.argmax(1))
                if maize_idx is not None:
                    p_maize.extend(probs[:, maize_idx].numpy().tolist())
            ys.append(y)

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(yhat).numpy()

    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="w" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); fig.tight_layout()
    cm_path = (args.save_prefix + "_confusion_matrix.png") if args.save_prefix else "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    print("saved:", cm_path)

    if C == 2 and maize_idx is not None:
        y_true_bin = (y_true == maize_idx).astype(int)
        p_m = np.array(p_maize)
        try:
            roc = roc_auc_score(y_true_bin, p_m)
            ap = average_precision_score(y_true_bin, p_m)
            print(f"ROC-AUC: {roc:.4f} | PR-AUC: {ap:.4f}")
            prec, rec, thr = precision_recall_curve(y_true_bin, p_m)
            plt.figure(figsize=(5,5))
            plt.plot(rec, prec)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (maize)")
            pr_path = (args.save_prefix + "_pr_curve.png") if args.save_prefix else "pr_curve.png"
            plt.tight_layout(); plt.savefig(pr_path, dpi=200)
            print("saved:", pr_path)
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
