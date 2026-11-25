"""Train a YOLO model for Document Layout Analysis (tables and charts).

Requirements:
    pip install ultralytics

Usage:
    python scripts/train_layout_yolo.py --data data/annotation_workspace/yolo/data.yaml --epochs 50 --batch 8
"""

import argparse
from pathlib import Path


def train_yolo(data_yaml: Path, model_name: str = "yolov8n.pt", epochs: int = 50, batch: int = 8, imgsz: int = 1024):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Please run:")
        print("  pip install ultralytics")
        return

    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    print(f"Starting training on {data_yaml} for {epochs} epochs...")
    model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=2,
        name="financial_layout_dla",
        save=True,
    )

    print("\nTraining completed! Evaluating model...")
    metrics = model.val()
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")
    print(f"Validation mAP50:    {metrics.box.map50:.4f}")

    print("\nExporting model to ONNX for CPU inference in Financial-Multimodal-RAG...")
    onnx_path = model.export(format="onnx", opset=12, dynamic=False, imgsz=imgsz)
    print(f"Exported ONNX model to: {onnx_path}")
    print("\nYou can configure this model in .env via:")
    print("  LAYOUT_BACKEND=yolo")
    print(f"  LAYOUT_MODEL_PATH={onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for table and chart detection.")
    parser.add_argument("--data", "-d", type=str, default="data/annotation_workspace/yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", "-m", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size (default: 1024)")
    args = parser.parse_args()

    train_yolo(Path(args.data), args.model, args.epochs, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
