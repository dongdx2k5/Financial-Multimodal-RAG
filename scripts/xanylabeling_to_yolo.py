"""Convert X-AnyLabeling / LabelMe JSON annotations into YOLO train/val dataset format."""

import argparse
import json
import random
import shutil
from pathlib import Path


def convert_xanylabeling_to_yolo(
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    val_split: float = 0.2,
) -> dict:
    categories = ["table", "chart"]
    cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for folder in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        folder.mkdir(parents=True, exist_ok=True)

    json_files = sorted(annotations_dir.glob("*.json"))
    random.seed(42)
    random.shuffle(json_files)
    num_val = int(len(json_files) * val_split)
    if val_split > 0 and len(json_files) >= 2 and num_val == 0:
        num_val = 1

    converted = 0
    box_counts = {cat: 0 for cat in categories}

    for idx, json_path in enumerate(json_files):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        img_name = Path(payload.get("imagePath") or f"{json_path.stem}.png").name
        src_img = images_dir / img_name
        if not src_img.is_file():
            continue
        img_w = float(payload.get("imageWidth") or 0)
        img_h = float(payload.get("imageHeight") or 0)
        if img_w <= 0 or img_h <= 0:
            continue

        is_val = idx < num_val
        dst_img_dir = val_img_dir if is_val else train_img_dir
        dst_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        lines = []
        for shape in payload.get("shapes", []):
            label = str(shape.get("label") or "").strip().lower()
            if label not in cat2idx:
                continue
            pts = shape.get("points") or []
            if len(pts) < 2:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            x0, x1 = max(0.0, min(xs)), min(img_w, max(xs))
            y0, y1 = max(0.0, min(ys)), min(img_h, max(ys))
            if x1 <= x0 or y1 <= y0:
                continue
            cx = (x0 + x1) / (2.0 * img_w)
            cy = (y0 + y1) / (2.0 * img_h)
            bw = (x1 - x0) / img_w
            bh = (y1 - y0) / img_h
            box_counts[label] += 1
            lines.append(f"{cat2idx[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        shutil.copy2(src_img, dst_img_dir / src_img.name)
        (dst_lbl_dir / f"{src_img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        converted += 1

    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: table\n"
        "  1: chart\n",
        encoding="utf-8",
    )
    return {"images": converted, "boxes": box_counts}


def main():
    parser = argparse.ArgumentParser(description="Convert X-AnyLabeling JSONs to YOLO dataset.")
    parser.add_argument("annotations_dir", type=Path, help="Folder containing X-AnyLabeling JSONs")
    parser.add_argument("--images-dir", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=Path("data/yolo_final"))
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    summary = convert_xanylabeling_to_yolo(
        args.annotations_dir, args.images_dir, args.output, args.val_split
    )
    print(f"Converted {summary['images']} images: {summary['boxes']}")


if __name__ == "__main__":
    main()
