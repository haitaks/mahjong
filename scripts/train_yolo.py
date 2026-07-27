"""Train YOLOv8 on the mahjong dataset.

Usage:
    python scripts/train_yolo.py                     # YOLOv8n, default params
    python scripts/train_yolo.py --model yolov8s     # YOLOv8s (more accurate)
    python scripts/train_yolo.py --epochs 50 --batch 16

After training, the best weights are at:
    yolo/runs/detect/train/weights/best.pt

Copy them back to the source machine:
    scp yolo/runs/detect/train/weights/best.pt user@source-machine:~/mahjong/yolo/model.pt
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "yolo" / "data.yaml"

if not DATA_YAML.exists():
    print(f"ERROR: {DATA_YAML} not found. Make sure you extracted the dataset:")
    print("  cd ~/mahjong && tar xzf yolo_dataset.tar.gz")
    sys.exit(1)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Train YOLO on mahjong dataset")
    p.add_argument("--model", default="yolov8n", help="YOLO model variant (yolov8n/s/m/l/x)")
    p.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    p.add_argument("--batch", type=int, default=8, help="Batch size (adjust to VRAM)")
    p.add_argument("--imgsz", type=int, default=640, help="Input image size")
    p.add_argument("--device", default="cuda", help="Device: cuda, cuda:0, cpu")
    p.add_argument("--data", default=str(DATA_YAML), help="Path to data.yaml")
    args = p.parse_args()

    print(f"Training {args.model} on {args.data}")
    print(f"  epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, device={args.device}")
    print()

    cmd = [
        sys.executable, "-c", f"""
from ultralytics import YOLO
model = YOLO('{args.model}.pt')
model.train(
    data=r'{args.data}',
    epochs={args.epochs},
    imgsz={args.imgsz},
    batch={args.batch},
    device=r'{args.device}',
    project='{ROOT / "yolo" / "runs"}',
    name='detect',
    exist_ok=True,
)
"""
    ]
    subprocess.run(cmd, check=True)

    weights = ROOT / "yolo" / "runs" / "detect" / "train" / "weights" / "best.pt"
    if weights.exists():
        print(f"\nTraining complete! Best weights: {weights}")
        print(f"\nCopy to source machine:")
        print(f"  scp {weights} user@source-machine:~/mahjong/yolo/model.pt")
    else:
        print("\nTraining finished but best.pt not found at expected path.")


if __name__ == "__main__":
    raise SystemExit(main())
