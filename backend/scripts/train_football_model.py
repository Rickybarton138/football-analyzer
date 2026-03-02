"""
Football Player Detection Model Fine-Tuning Script

Downloads the Roboflow football-players-detection dataset (9,068 images, 4 classes:
ball, goalkeeper, player, referee) and fine-tunes a YOLO model at imgsz=1280
with football-appropriate augmentation.

Output: data/models/football_best.pt

Usage:
    python -m scripts.train_football_model
    python -m scripts.train_football_model --base-model yolo11m.pt --epochs 50
"""
import argparse
import sys
from pathlib import Path


def download_dataset(api_key: str, output_dir: Path) -> Path:
    """Download Roboflow football-players-detection dataset."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(1)

    dataset = version.download("yolov8", location=str(output_dir / "football-players"))
    print(f"[TRAIN] Dataset downloaded to: {dataset.location}")
    return Path(dataset.location)


def train(
    dataset_path: Path,
    base_model: str = "yolo11m.pt",
    epochs: int = 80,
    imgsz: int = 1280,
    batch_size: int = 8,
    output_dir: Path = Path("data/models"),
):
    """Fine-tune YOLO on football dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    print(f"[TRAIN] Base model: {base_model}")
    print(f"[TRAIN] Dataset: {data_yaml}")
    print(f"[TRAIN] Epochs: {epochs}, ImgSz: {imgsz}, Batch: {batch_size}")

    model = YOLO(base_model)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        # Football-appropriate augmentation
        mosaic=1.0,          # Mosaic augmentation (multiple images combined)
        mixup=0.1,           # MixUp augmentation (image blending)
        hsv_h=0.015,         # HSV hue variation (lighting changes)
        hsv_s=0.7,           # HSV saturation variation
        hsv_v=0.4,           # HSV value variation
        degrees=5.0,         # Small rotation (camera tilt)
        translate=0.1,       # Translation
        scale=0.5,           # Scale variation (players at different distances)
        fliplr=0.5,          # Horizontal flip
        flipud=0.0,          # No vertical flip (not realistic for football)
        # Training params
        patience=20,         # Early stopping patience
        save=True,
        save_period=10,      # Save checkpoint every 10 epochs
        device="0",          # GPU 0 (change to "cpu" if no GPU)
        workers=4,
        project=str(output_dir / "training_runs"),
        name="football_finetune",
        exist_ok=True,
    )

    # Copy best model to standard location
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    output_path = output_dir / "football_best.pt"

    if best_model_path.exists():
        import shutil
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model_path, output_path)
        print(f"[TRAIN] Best model saved to: {output_path}")

        # Validate
        model = YOLO(str(output_path))
        val_results = model.val(data=str(data_yaml), imgsz=imgsz)
        print(f"[TRAIN] Validation mAP50: {val_results.box.map50:.4f}")
        print(f"[TRAIN] Validation mAP50-95: {val_results.box.map:.4f}")
    else:
        print(f"[TRAIN] WARNING: Best model not found at {best_model_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for football player detection")
    parser.add_argument("--api-key", type=str, help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--base-model", type=str, default="yolo11m.pt", help="Base YOLO model to fine-tune")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--dataset-dir", type=str, default="data/datasets", help="Dataset download directory")
    parser.add_argument("--output-dir", type=str, default="data/models", help="Model output directory")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download (use existing)")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    dataset_path = dataset_dir / "football-players"

    if not args.skip_download:
        if not api_key:
            print("ERROR: Roboflow API key required. Use --api-key or set ROBOFLOW_API_KEY")
            sys.exit(1)
        dataset_path = download_dataset(api_key, dataset_dir)
    else:
        if not dataset_path.exists():
            print(f"ERROR: Dataset not found at {dataset_path}. Remove --skip-download to download it.")
            sys.exit(1)

    output_path = train(
        dataset_path=dataset_path,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        output_dir=output_dir,
    )

    print(f"\n[TRAIN] Done! Model saved to: {output_path}")
    print(f"[TRAIN] Add to config.py YOLO_MODEL_CHAIN to use automatically.")


if __name__ == "__main__":
    main()
