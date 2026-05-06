"""
RCDyno_Infer.py
---------------
Loads a trained RCDyno checkpoint and runs anomaly detection on one
image, a folder of images, or the entire test dataset.

Usage:
    python RCDyno_Infer.py

Edit the CONFIG section below to point at a specific image/folder,
or leave IMAGE_PATH as None to run against every image in dataset/test/.

Output:
    - Console verdict per image  (NORMAL / ANOMALOUS + score)
    - Visualisation images saved to inference_output/
"""

from pathlib import Path
import sys

from anomalib.engine import Engine
from anomalib.models import EfficientAd

# =============================================================================
# Configuration
# =============================================================================

# Path to a single image OR a folder of images.
# Set to None to infer on the entire dataset/test/ tree.
IMAGE_PATH: Path | None = None

# Checkpoint to use. None = auto-select the most recently trained one.
CKPT_PATH: Path | None = None

# Directories
DATASET_ROOT = Path(__file__).parent / "dataset"
RESULTS_DIR  = Path(__file__).parent / "anomalib_results"
OUTPUT_DIR   = Path(__file__).parent / "inference_output"

# Anomaly score threshold: scores above this are flagged as ANOMALOUS.
# None = use the threshold stored in the checkpoint (recommended).
SCORE_THRESHOLD: float | None = None

# Model class must match what was used in training
MODEL_CLASS = EfficientAd


# =============================================================================
# Helpers
# =============================================================================

def find_latest_checkpoint(results_dir: Path) -> Path:
    """Return the most recently modified .ckpt file under results_dir."""
    ckpts = sorted(results_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint (.ckpt) files found under: {results_dir}\n"
            "Train a model first with RCDyno_Train.py."
        )
    return ckpts[-1]


def collect_images(path: Path) -> list[Path]:
    """Return a sorted list of all PNG/JPG images at path (file or folder)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    if path.is_file():
        return [path]
    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found under: {path}")
    return images


def print_header(ckpt_path: Path, image_path: Path, output_dir: Path):
    print()
    print("=" * 60)
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Input       : {image_path}")
    print(f"  Output dir  : {output_dir}")
    print("=" * 60)


# =============================================================================
# Inference
# =============================================================================

def run_inference(image_path: Path, ckpt_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print_header(ckpt_path, image_path, output_dir)

    model = MODEL_CLASS()

    engine = Engine(default_root_dir=output_dir)

    predictions = engine.predict(
        model=model,
        data_path=image_path,
        ckpt_path=ckpt_path,
    )

    if not predictions:
        print("No predictions returned.")
        return

    print(f"\nProcessed {sum(len(b.image_path or []) for b in predictions)} image(s).\n")

    normal_count   = 0
    anomalous_count = 0

    for batch in predictions:
        paths  = batch.image_path or []
        scores = batch.pred_score
        labels = batch.pred_label

        scores = scores.tolist() if hasattr(scores, "tolist") else (scores or [])
        labels = labels.tolist() if hasattr(labels, "tolist") else (labels or [])

        for i, path in enumerate(paths):
            score = scores[i] if i < len(scores) else None
            label = labels[i] if i < len(labels) else None

            # Apply manual threshold if configured
            if SCORE_THRESHOLD is not None and score is not None:
                label = score > SCORE_THRESHOLD

            verdict   = "ANOMALOUS" if label else "NORMAL"
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)

            # Colour-code using ANSI (works in most terminals)
            color     = "\033[91m" if label else "\033[92m"  # red / green
            reset     = "\033[0m"

            # Show the subfolder name as context (e.g. fire, good, …)
            rel = Path(path)
            try:
                rel = rel.relative_to(DATASET_ROOT)
            except ValueError:
                pass

            print(f"  {color}[{verdict:9s}]{reset}  score={score_str}  {rel}")

            if label:
                anomalous_count += 1
            else:
                normal_count += 1

    total = normal_count + anomalous_count
    print()
    print("-" * 60)
    print(f"  NORMAL    : {normal_count:>4}  ({100*normal_count/total:.1f}%)" if total else "")
    print(f"  ANOMALOUS : {anomalous_count:>4}  ({100*anomalous_count/total:.1f}%)" if total else "")
    print("-" * 60)
    print(f"  Visualisations saved to: {output_dir.resolve()}")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    # Resolve checkpoint
    ckpt_path = CKPT_PATH if CKPT_PATH else find_latest_checkpoint(RESULTS_DIR)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    # Resolve image source
    if IMAGE_PATH is not None:
        infer_path = Path(IMAGE_PATH)
    else:
        infer_path = DATASET_ROOT / "test"
        if not infer_path.is_dir():
            sys.exit(
                f"No IMAGE_PATH set and default test folder not found: {infer_path}\n"
                "Either set IMAGE_PATH or run Create_Test_Data.py first."
            )

    if not infer_path.exists():
        sys.exit(f"Image path not found: {infer_path}")

    run_inference(
        image_path=infer_path,
        ckpt_path=ckpt_path,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
