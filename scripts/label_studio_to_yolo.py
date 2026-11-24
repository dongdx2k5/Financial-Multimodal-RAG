"""Converts Label Studio JSON export to YOLO object detection format.

Usage:
    python scripts/label_studio_to_yolo.py export.json --images-dir data/annotation_workspace/images --output data/yolo_final
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def convert_label_studio_to_yolo(export_json_path: Path, images_dir: Path, output_dir: Path, val_split: float = 0.2):
    categories = ["table", "chart"]
    cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    data = json.loads(export_json_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} tasks from {export_json_path.name}")

    converted_count = 0
    box_counts = {cat: 0 for cat in categories}

    # Deterministic shuffle for reproducibility
    random.seed(42)
    random.shuffle(data)
    num_val = int(len(data) * val_split)

    for idx, item in enumerate(data):
        is_val = idx < num_val
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        # Identify image filename
        img_info = item.get("data", {})
        raw_image_path = img_info.get("image", "") or img_info.get("filename", "")
        img_filename = Path(raw_image_path.split("?d=")[-1] if "?d=" in raw_image_path else raw_image_path).name

        src_image = images_dir / img_filename
        if not src_image.exists():
            # Try searching in images_dir
            candidates = list(images_dir.glob(f"*{img_filename}"))
            if candidates:
                src_image = candidates[0]
            else:
                continue

        # Collect annotations (from manual annotations or accepted predictions)
        annotations_list = item.get("annotations", [])
        if not annotations_list and item.get("predictions"):
            annotations_list = item.get("predictions")

        lines = []
        for ann in annotations_list:
            for result in ann.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                val = result.get("value", {})
                labels = val.get("rectanglelabels", [])
                if not labels:
                    continue
                label = labels[0].lower()
                if label not in cat2idx:
                    continue

                class_id = cat2idx[label]
                box_counts[label] += 1

                # Label Studio percentages (0-100) -> YOLO (0-1)
                x = val.get("x", 0) / 100.0
                y = val.get("y", 0) / 100.0
                w = val.get("width", 0) / 100.0
                h = val.get("height", 0) / 100.0

                cx = x + w / 2.0
                cy = y + h / 2.0

                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Copy image and write label file
        stem = src_image.stem
        shutil.copy2(src_image, target_img_dir / src_image.name)
        (target_lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        converted_count += 1

    # Write data.yaml
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: table\n"
        "  1: chart\n"
    )
    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print("\nConversion Completed!")
    print(f"Total processed images: {converted_count}")
    print(f"Boxes extracted: {box_counts}")
    print(f"Output dataset: {output_dir.resolve()}")
    print(f"Ready for YOLO training with data.yaml at '{output_dir / 'data.yaml'}'")


def main():
    parser = argparse.ArgumentParser(description="Convert Label Studio export to YOLO format.")
    parser.add_argument("export_json", type=str, help="Path to exported Label Studio JSON file")
    parser.add_argument("--images-dir", "-i", type=str, required=True, help="Directory containing original images")
    parser.add_argument("--output", "-o", type=str, default="data/yolo_final", help="Output directory for YOLO dataset")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    args = parser.parse_args()

    convert_label_studio_to_yolo(Path(args.export_json), Path(args.images_dir), Path(args.output), args.val_split)


if __name__ == "__main__":
    main()
