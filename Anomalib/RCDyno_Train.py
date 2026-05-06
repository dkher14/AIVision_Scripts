"""
RCDyno_Train.py
---------------
Trains an anomaly detection model on the local RCDyno dataset.

Expected folder structure (produced by Create_Training_Data.py):

    dataset/
        train/
            good/       ← normal images used for training
        test/           ← optional; used for evaluation if present
            good/
            fire/
            horizontal_displacement/
            vertical_displacement/

Usage:
    python RCDyno_Train.py

The trained checkpoint is saved under anomalib_results/ and is
automatically picked up by RCDyno_Infer.py.
"""

import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

from pathlib import Path
from time import perf_counter

import torch
torch.set_float32_matmul_precision("high")

from torchvision.transforms import v2 as T

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Patchcore

# =============================================================================
# Configuration
# =============================================================================
DATASET_ROOT     = Path(__file__).parent / "dataset"
RESULTS_DIR      = Path(__file__).parent / "anomalib_results"

# Model to use: "efficientad" | "patchcore"
MODEL_NAME       = "efficientad"

IMAGE_SIZE       = 256
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE  = 8
NUM_WORKERS      = 4
MAX_EPOCHS       = 30
VAL_SPLIT_RATIO  = 0.1   # fraction of train/good used for validation


# =============================================================================
# Helpers
# =============================================================================

def build_model(name: str):
    n = name.lower()
    if n == "efficientad":
        return EfficientAd()
    if n == "patchcore":
        return Patchcore()
    raise ValueError(f"Unknown model '{name}'. Choose from: efficientad, patchcore")


def has_test_data() -> bool:
    """Returns True if at least one test subfolder with images exists."""
    test_dir = DATASET_ROOT / "test"
    if not test_dir.is_dir():
        return False
    for sub in test_dir.iterdir():
        if sub.is_dir() and any(sub.glob("*.png")):
            return True
    return False


# =============================================================================
# Main training routine
# =============================================================================

def main():
    overall_start = perf_counter()

    train_dir = DATASET_ROOT / "train" / "good"
    if not train_dir.is_dir() or not any(train_dir.glob("*.png")):
        raise FileNotFoundError(
            f"No training images found in: {train_dir}\n"
            "Run Create_Training_Data.py first to capture training images."
        )

    test_dir = DATASET_ROOT / "test" if has_test_data() else None

    print()
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME.upper()}")
    print(f"  Dataset    : {DATASET_ROOT.resolve()}")
    print(f"  Test data  : {'yes' if test_dir else 'not found — skipping evaluation'}")
    print(f"  Results    : {RESULTS_DIR.resolve()}")
    print("=" * 60)

    resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))

    # Folder datamodule — works with any custom MVTec-style directory
    datamodule = Folder(
        name="rcdyno",
        root=DATASET_ROOT,
        normal_dir="train/good",
        # If test data exists point to test/good as the "normal" test split;
        # all other test subfolders are treated as anomalous automatically.
        normal_test_dir="test/good" if test_dir else None,
        abnormal_dir="test" if test_dir else None,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        augmentations=resize,
        val_split_mode="from_train",
        val_split_ratio=VAL_SPLIT_RATIO,
    )

    model = build_model(MODEL_NAME)

    engine = Engine(
        default_root_dir=RESULTS_DIR,
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
    )

    print("\nStarting training …")
    train_start = perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    train_elapsed = perf_counter() - train_start

    print(f"Training time: {train_elapsed:.2f} seconds ({train_elapsed / 60:.2f} minutes)")

    if test_dir:
        print("\nRunning evaluation on test set …")
        test_results = engine.test(model=model, datamodule=datamodule)

        print("\nTest results:")
        for result in test_results:
            for key, value in result.items():
                print(
                    f"  {key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"  {key}: {value}"
                )
    else:
        print("\nNo test data found — skipping evaluation.")

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Checkpoint saved under: {RESULTS_DIR.resolve()}")
    total_elapsed = perf_counter() - overall_start
    print(f"  Training time         : {train_elapsed:.2f} s ({train_elapsed / 60:.2f} min)")
    print(f"  Total script runtime  : {total_elapsed:.2f} s ({total_elapsed / 60:.2f} min)")
    print("  Run RCDyno_Infer.py to test on new images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
