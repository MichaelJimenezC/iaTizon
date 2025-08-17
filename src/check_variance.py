import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import imagehash

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt


def iter_images(root: Path, splits=("train", "valid", "test"),
                exts=(".jpg",".jpeg",".png",".bmp",".webp")):
    for split in splits:
        d = root / split
        if not d.exists():
            continue
        for cls_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
            for f in cls_dir.iterdir():
                if f.suffix.lower() in exts:
                    yield split, cls_dir.name, f


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


@torch.no_grad()
def extract_embeddings(files, device="cpu", batch_size=32, img_size=224):
    """
    Extrae embeddings (512-D) usando ResNet-18 preentrenada (penúltima capa).
    """
    # Modelo
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(m.children())[:-1]).to(device).eval()  # hasta GAP (512x1x1)

    tfm = build_transform(img_size)
    embs = []
    meta = []

    batch_imgs = []
    batch_meta = []
    for (split, cls, f) in files:
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            # imagen corrupta
            continue
        x = tfm(img)
        batch_imgs.append(x)
        batch_meta.append((split, cls, str(f)))
        if len(batch_imgs) == batch_size:
            X = torch.stack(batch_imgs).to(device)
            z = backbone(X).squeeze(-1).squeeze(-1)  # (B, 512)
            embs.append(z.cpu().numpy())
            meta.extend(batch_meta)
            batch_imgs, batch_meta = [], []

    if batch_imgs:
        X = torch.stack(batch_imgs).to(device)
        z = backbone(X).squeeze(-1).squeeze(-1)
        embs.append(z.cpu().numpy())
        meta.extend(batch_meta)

    if embs:
        embs = np.concatenate(embs, axis=0)
    else:
        embs = np.zeros((0,512), dtype=np.float32)
    return embs, meta


def compute_phash_dups(files, max_pairs=20000, ham_thresh=5):
    """
    Detecta casi-duplicados por perceptual hash.
    Devuelve conteo y ejemplos por par de splits (p.ej., train-valid).
    """
    # Guardamos hashes por (split, cls)
    phashes = []
    for (split, cls, f) in files:
        try:
            img = Image.open(f).convert("RGB")
            h = imagehash.phash(img)  # 64-bit
            phashes.append((split, cls, str(f), h))
        except Exception:
            continue

    # Índices por split para comparar entre splits (evitar O(n^2) total)
    by_split = defaultdict(list)
    for s, c, fp, h in phashes:
        by_split[s].append((c, fp, h))

    pairs = [("train","valid"), ("train","test"), ("valid","test")]
    results = {}
    for a,b in pairs:
        if a not in by_split or b not in by_split:
            continue
        A = by_split[a]
        B = by_split[b]
        cnt = 0
        examples = []
        # Submuestreo simple si hay demasiadas combinaciones
        stepA = max(1, len(A) // int(np.sqrt(max_pairs)))
        stepB = max(1, len(B) // int(np.sqrt(max_pairs)))
        for i in range(0, len(A), stepA):
            for j in range(0, len(B), stepB):
                ha = A[i][2]; hb = B[j][2]
                ham = ha - hb  # distancia Hamming
                if ham <= ham_thresh:
                    cnt += 1
                    if len(examples) < 20:
                        examples.append((A[i][1], B[j][1], ham, A[i][0], B[j][0]))
        results[(a,b)] = (cnt, examples)
    return results


def summarize_variance(embs, meta):
    """
    Métricas de diversidad:
    - Distancia intra-clase promedio (coseno)
    - Distancia inter-clase promedio (coseno)
    - Silhouette score (por split y global si aplica)
    - Varianza por clase (traza de covarianza)
    """
    if len(embs) == 0:
        return {}

    X = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    labels = np.array([m[1] for m in meta])
    splits = np.array([m[0] for m in meta])

    classes = sorted(np.unique(labels))

    metrics = {"per_split": {}}

    def _compute_for(mask, tag):
        _X = X[mask]
        _y = labels[mask]
        if len(_X) < 10 or len(np.unique(_y)) < 2:
            return None

        # pairwise cos distances
        D = pairwise_distances(_X, metric="cosine")
        # intra/inter
        intra = []
        inter = []
        for c in np.unique(_y):
            idx_c = np.where(_y == c)[0]
            idx_nc = np.where(_y != c)[0]
            if len(idx_c) > 1:
                Dc = D[np.ix_(idx_c, idx_c)]
                intra.append(Dc[np.triu_indices_from(Dc, k=1)].mean())
            if len(idx_c) > 0 and len(idx_nc) > 0:
                Dcn = D[np.ix_(idx_c, idx_nc)].mean()
                inter.append(Dcn)
        intra_m = float(np.mean(intra)) if intra else float("nan")
        inter_m = float(np.mean(inter)) if inter else float("nan")

        # silhouette
        try:
            sil = float(silhouette_score(_X, _y, metric="cosine"))
        except Exception:
            sil = float("nan")

        # varianza por clase (traza cov)
        var_trace = {}
        for c in np.unique(_y):
            idx_c = np.where(_y == c)[0]
            if len(idx_c) > 1:
                cov = np.cov(_X[idx_c].T)
                var_trace[c] = float(np.trace(cov))
            else:
                var_trace[c] = float("nan")

        return {
            "n_samples": int(len(_X)),
            "intra_cosine_mean": intra_m,
            "inter_cosine_mean": inter_m,
            "silhouette_cosine": sil,
            "var_trace_per_class": var_trace,
        }

    # Por split
    for s in sorted(np.unique(splits)):
        res = _compute_for(splits == s, s)
        if res: metrics["per_split"][s] = res

    # Global
    res_global = _compute_for(np.ones(len(X), dtype=bool), "global")
    if res_global: metrics["global"] = res_global

    return metrics


def plot_pca(embs, meta, out_path):
    if len(embs) == 0:
        return
    X = embs
    labels = np.array([m[1] for m in meta])
    splits = np.array([m[0] for m in meta])
    classes = sorted(np.unique(labels))
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    # Un plot por split
    for s in sorted(np.unique(splits)):
        mask = (splits == s)
        if mask.sum() < 2: 
            continue
        plt.figure(figsize=(6,5))
        for c in classes:
            msc = mask & (labels == c)
            if msc.sum() == 0: 
                continue
            plt.scatter(Z[msc,0], Z[msc,1], s=10, label=c, alpha=0.7)
        plt.title(f"PCA 2D embeddings - split: {s}")
        plt.legend(markerscale=2, fontsize=8)
        plt.tight_layout()
        fp = os.path.join(out_path, f"pca_{s}.png")
        plt.savefig(fp, dpi=150)
        plt.close()

    # Plot global
    plt.figure(figsize=(6,5))
    for c in classes:
        msc = (labels == c)
        if msc.sum() == 0: 
            continue
        plt.scatter(Z[msc,0], Z[msc,1], s=10, label=c, alpha=0.7)
    plt.title("PCA 2D embeddings - global")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    fp = os.path.join(out_path, f"pca_global.png")
    plt.savefig(fp, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Raíz del dataset con carpetas train/valid/test y subcarpetas por clase")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit_per_class", type=int, default=0,
                    help="Máximo de imágenes por clase+split (0 = sin límite)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--outdir", type=str, default="variance_report")
    ap.add_argument("--phash_hamming", type=int, default=5,
                    help="Umbral Hamming para considerar casi-duplicado entre splits")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Recolecta lista de archivos (con límite por clase+split si se pide)
    per_key = defaultdict(list)  # key = (split, cls)
    for split, cls, f in iter_images(root):
        per_key[(split, cls)].append((split, cls, f))

    files = []
    for k, lst in per_key.items():
        if args.limit_per_class and len(lst) > args.limit_per_class:
            lst = lst[:args.limit_per_class]
        files.extend(lst)

    # ====== Embeddings ======
    print(f"[INFO] Extrayendo embeddings de {len(files)} imágenes en {args.device} ...")
    embs, meta = extract_embeddings(files, device=args.device,
                                    batch_size=args.batch_size, img_size=args.img_size)
    print(f"[OK] Embeddings: {embs.shape}")

    # ====== Métricas de varianza ======
    metrics = summarize_variance(embs, meta)
    print("\n=== MÉTRICAS DE VARIANZA (cosine) ===")
    def pretty(d, indent=0):
        for k,v in d.items():
            if isinstance(v, dict):
                print("  "*indent + f"{k}:")
                pretty(v, indent+1)
            else:
                print("  "*indent + f"{k}: {v}")

    pretty(metrics)

    # Heurísticas simples de interpretación
    def interpret(m):
        if not m or "global" not in m:
            return
        g = m["global"]
        intra = g["intra_cosine_mean"]
        inter = g["inter_cosine_mean"]
        sil = g["silhouette_cosine"]
        print("\n=== INTERPRETACIÓN HEURÍSTICA ===")
        print(f"- Intra-clase (cosine) promedio: {intra:.3f}  (menor es mejor; <0.3 suele indicar clases compactas)")
        print(f"- Inter-clase (cosine) promedio: {inter:.3f}  (mayor es mejor; >0.5–0.7 indica buena separación)")
        print(f"- Silhouette (cosine): {sil:.3f}  (~0.2–0.5 ok, >0.5 muy bueno; negativo = clases mezcladas)")
        if np.isfinite(intra) and np.isfinite(inter):
            ratio = inter / (intra + 1e-9)
            print(f"- Ratio inter/intra: {ratio:.2f}  (>1.5 suele ser saludable)")
    interpret(metrics)

    # ====== PCA plots ======
    print("\n[INFO] Generando PCA plots ...")
    plot_pca(embs, meta, str(outdir))
    print(f"[OK] Guardado: {outdir / 'pca_*.png'}")

    # ====== Detección de casi-duplicados entre splits ======
    print("\n[INFO] Buscando casi-duplicados (phash) entre splits ...")
    # Para ahorrar tiempo, usa los mismos archivos ya listados
    phash_results = compute_phash_dups(files, ham_thresh=args.phash_hamming)
    for (a,b), (cnt, examples) in phash_results.items():
        print(f"- {a} vs {b}: {cnt} posibles casi-duplicados (Hamming <= {args.phash_hamming})")
        for i,(fa, fb, ham, ca, cb) in enumerate(examples):
            print(f"   ej{i+1}: {fa}  <->  {fb}  | ham={ham} | {ca} vs {cb}")

    # ====== Resumen por conteo de clases ======
    print("\n=== CONTEO POR CLASE Y SPLIT ===")
    counts = Counter((m[0], m[1]) for m in meta)
    by_split = defaultdict(dict)
    for (s,c), n in counts.items():
        by_split[s][c] = n
    for s in sorted(by_split.keys()):
        print(f"[{s}]")
        for c in sorted(by_split[s].keys()):
            print(f"  {c:20s}: {by_split[s][c]}")

    print(f"\nListo. Revisa los PNG en: {outdir.resolve()}")
    print("Interpretación rápida:")
    print("- Silhouette ~0.2–0.5 y ratio inter/intra >1.5 ⇒ varianza/ separación razonables.")
    print("- Muchos casi-duplicados entre splits ⇒ posible fuga; re-haz los splits o depura duplicados.")
    print("- PCA con nubes muy apretadas dentro de cada clase + inter grande ⇒ buena diversidad controlada.")
    print("- PCA con nubes muy compactas y separadas, pero val/test perfecto sospechoso ⇒ revisa fugas/escenarios.")
    

if __name__ == "__main__":
    main()
