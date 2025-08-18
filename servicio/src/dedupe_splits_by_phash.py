import argparse, csv, os, shutil
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import imagehash

EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def scan_files(root: Path, splits=("train","valid","test")):
    files = []
    for s in splits:
        d = root / s
        if not d.exists(): 
            continue
        for cls_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
            for f in cls_dir.rglob("*"):
                if f.suffix.lower() in EXTS and f.is_file():
                    files.append((s, cls_dir.name, f))
    return files

def compute_phashes(files):
    rows = []
    for s, c, f in files:
        try:
            img = Image.open(f).convert("RGB")
            h = imagehash.phash(img)  # 64-bit hash
            rows.append({"split":s, "class":c, "path":str(f), "phash":str(h)})
        except Exception as e:
            print(f"[WARN] No se pudo hashear: {f} ({e})")
    return rows

def group_collisions(rows, ham_thresh):
    # Agrupa por hash aproximado: comparamos todos-vs-todos por hash textual
    # Optimización simple: agrupar por los primeros 8 hex para reducir comparaciones
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["phash"][:8]].append(r)

    collisions = []
    for key, lst in buckets.items():
        n = len(lst)
        if n < 2: 
            continue
        # comparar pares dentro del bucket
        for i in range(n):
            for j in range(i+1, n):
                ha = imagehash.hex_to_hash(lst[i]["phash"])
                hb = imagehash.hex_to_hash(lst[j]["phash"])
                ham = ha - hb
                if ham <= ham_thresh:
                    a, b = lst[i], lst[j]
                    # normaliza el orden (split priority)
                    collisions.append((a, b, ham))
    return collisions

def solve_sets(collisions, prefer=("train","valid","test")):
    """Construye grupos por archivo y decide qué conservar según prioridad."""
    # Build disjoint-set over files: cualquier colisión conecta archivos
    parent = {}
    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return parent.get(x, x)
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    files = set()
    for a,b,_ in collisions:
        pa, pb = a["path"], b["path"]
        files.update([pa, pb])
        if pa not in parent: parent[pa] = pa
        if pb not in parent: parent[pb] = pb
        union(pa, pb)

    groups = defaultdict(list)
    for a,b,ham in collisions:
        ra = find(a["path"])
        rb = find(b["path"])
        groups[find(a["path"])].append((a,b,ham))

    # Recolectar miembros únicos por grupo
    members = defaultdict(dict)  # root -> {path: meta}
    for r in set(parent.keys()):
        root = find(r)
        # buscar meta (split, class) en collisions para este path
        # para no recorrer mucho, hacemos un índice:
    idx = {}
    for a,b,ham in collisions:
        idx[a["path"]] = a
        idx[b["path"]] = b
    for fp in parent.keys():
        root = find(fp)
        members[root][fp] = idx[fp]

    decisions = []
    for root, m in members.items():
        # elegir keeper por prioridad de split
        keep = None
        # ordenar candidatos por prefer y luego por clase para estabilidad
        ordered = sorted(m.values(), key=lambda r: (prefer.index(r["split"]) if r["split"] in prefer else 999, r["class"], r["path"]))
        if ordered:
            keep = ordered[0]
        to_move = [r for r in ordered[1:]]
        decisions.append((keep, to_move))
    return decisions

def move_to_quarantine(root: Path, quarantine: Path, to_move, dry=True):
    moved = []
    for r in to_move:
        src = Path(r["path"])
        # Estructura: _quarantine/<split>/<class>/<filename>
        dst = quarantine / r["split"] / r["class"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dry:
            moved.append((src, dst))
        else:
            try:
                shutil.move(str(src), str(dst))
                moved.append((src, dst))
            except Exception as e:
                print(f"[ERR] No pude mover {src} -> {dst}: {e}")
    return moved

def write_report(csv_path, collisions, decisions, moved_preview):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type","split_a","class_a","path_a","split_b","class_b","path_b","hamming"])
        for a,b,ham in collisions:
            w.writerow(["collision", a["split"], a["class"], a["path"], b["split"], b["class"], b["path"], ham])
        w.writerow([])
        w.writerow(["decision","keep_split","keep_class","keep_path","n_removed"])
        for keep, to_move in decisions:
            w.writerow(["keep", keep["split"], keep["class"], keep["path"], len(to_move)])
        w.writerow([])
        w.writerow(["moved_preview","src","dst"])
        for src,dst in moved_preview:
            w.writerow(["moved", str(src), str(dst)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Raíz del dataset con train/valid/test")
    ap.add_argument("--ham", type=int, default=0, help="Umbral Hamming (0=exactos, 5=casi duplicados)")
    ap.add_argument("--quarantine", default="_quarantine", help="Carpeta destino para mover duplicados")
    ap.add_argument("--fix", action="store_true", help="Si se pasa, mueve archivos (si no, dry-run)")
    ap.add_argument("--prefer", default="train,valid,test", help="Prioridad de conservación (coma-separada)")
    args = ap.parse_args()

    root = Path(args.root)
    quarantine = Path(args.quarantine) if Path(args.quarantine).is_absolute() else (root / args.quarantine)
    prefer = tuple([s.strip() for s in args.prefer.split(",")])

    print(f"[SCAN] Buscando imágenes en {root} ...")
    files = scan_files(root)
    print(f"[SCAN] {len(files)} archivos encontrados.")

    print("[HASH] Calculando perceptual hashes ...")
    rows = compute_phashes(files)
    print(f"[HASH] {len(rows)} hasheados.")

    print(f"[MATCH] Buscando colisiones con Hamming <= {args.ham} ...")
    collisions = group_collisions(rows, ham_thresh=args.ham)
    print(f"[MATCH] {len(collisions)} pares en conflicto.")

    if not collisions:
        print("✅ No se detectaron duplicados entre splits con ese umbral. Nada que hacer.")
        return

    decisions = solve_sets(collisions, prefer=prefer)

    total_to_move = sum(len(tm) for _, tm in decisions)
    print(f"[PLAN] Se conservarán {len(decisions)} representativos. {total_to_move} archivos serán movidos a: {quarantine}")
    preview = []
    for keep, to_move in decisions:
        moved = move_to_quarantine(root, quarantine, to_move, dry=not args.fix)
        preview.extend(moved)

    report_csv = root / "dedupe_report.csv"
    write_report(report_csv, collisions, decisions, preview)
    print(f"[REPORT] CSV: {report_csv}")

    if args.fix:
        print("✅ Hecho: duplicados movidos a cuarentena.")
    else:
        print("👀 Dry-run: no se movió nada. Ejecuta con --fix para aplicar cambios.")

if __name__ == "__main__":
    main()
