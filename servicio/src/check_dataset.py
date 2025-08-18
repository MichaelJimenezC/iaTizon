from pathlib import Path

root = Path(r"C:\dev\dataset")  
for split in ["train", "valid", "test"]:
    d = root / split
    if not d.exists(): 
        print(f"[{split}] no existe"); 
        continue
    print(f"\n[{split}]")
    for cls_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
        n = len([f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]])
        print(f"  {cls_dir.name:12s}: {n} imágenes")
