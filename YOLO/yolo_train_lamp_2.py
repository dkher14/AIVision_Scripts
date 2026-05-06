# Added customization to reduce background dependency.
from ultralytics import YOLO

def build_custom_albumentations(imgsz: int):
    """
    Additional Albumentations transforms to reduce background dependency.
    - Blur/Noise: Add robustness to motion blur and sensor noise (low probability).
    - Brightness/Contrast/Gamma: Improve tolerance to reflections and exposure differences.
    - CoarseDropout: Suppress over-reliance on background (keep probability low).
    Note: With the Ultralytics Python API, you can pass custom Albumentations via
          model.train(..., augmentations=[...]). This replaces the default Albumentations. [2](https://githubissues.com/ultralytics/ultralytics/20124)[1](https://docs.ultralytics.com/modes/train/)
    """
    import albumentations as A

    # Scale dropout region by image size (avoid removing small objects too aggressively)
    dropout_max = max(24, imgsz // 24)  # For imgsz=512, ~21 -> rounded up to 24
    dropout_max = min(dropout_max, 48)

    transforms = [
        # Blur (low probability)
        A.OneOf(
            [
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ],
            p=0.10,
        ),

        # Noise (low probability)
        A.OneOf(
            [
                A.GaussNoise(var_limit=(10.0, 60.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.6), p=1.0),
            ],
            p=0.10,
        ),

        # Become more robust to reflections/exposure differences (medium to low probability)
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            brightness_by_max=True,
            p=0.15,
        ),
        A.RandomGamma(
            gamma_limit=(80, 130),
            p=0.15,
        ),

        # Local contrast enhancement (low probability)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.05),

        # Partial occlusion (low probability: avoid overdoing it)
        A.CoarseDropout(
            max_holes=8,
            max_height=dropout_max,
            max_width=dropout_max,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.07,
        ),
    ]
    return transforms


def main():
    model = YOLO("yolov8s.pt")

    imgsz = 512

    # Build Albumentations (additional augmentations)
    custom_transforms = build_custom_albumentations(imgsz)

    model.train(
        data="lamp.yaml",
        epochs=100,
        imgsz=imgsz,         # Start with a lighter setup first
        batch=8,             # If you hit OOM, reduce to 4
        workers=2,           # Compare 0/2/4 to see what’s fastest
        project="runs",
        name="lamp_20260106_bgbreak",
        exist_ok=True,

        # -----------------------------
        # Lighten outputs to check speed (keeping your approach)
        # -----------------------------
        val=True,
        plots=False,

        # -----------------------------
        # Break background dependency: mix-based augmentations (Ultralytics built-in)
        # -----------------------------
        mosaic=1.0,          # Strongly disrupt background consistency [1](https://docs.ultralytics.com/modes/train/)
        mixup=0.10,          # Enable slightly (don’t overdo it) [1](https://docs.ultralytics.com/modes/train/)
        cutmix=0.10,         # Enable slightly (don’t overdo it) [1](https://docs.ultralytics.com/modes/train/)
        close_mosaic=10,     # Disable mosaic near the end to match real distribution (optional) [1](https://docs.ultralytics.com/modes/train/)

        # -----------------------------
        # Improve robustness to lighting differences (reflections, overexposure, low light): HSV
        # -----------------------------
        hsv_h=0.015,
        hsv_s=0.60,
        hsv_v=0.50,          # Bias toward brightness variation (increase slightly if reflections are troublesome) [1](https://docs.ultralytics.com/modes/train/)

        # -----------------------------
        # Geometric transforms (too strong can be harmful for meter-like objects, so keep small)
        # -----------------------------
        degrees=2.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,     # Recommend OFF initially (if needed, try ~0.0005 and observe) [1](https://docs.ultralytics.com/modes/train/)
        fliplr=0.0,          # Horizontal flip can cause issues for meter-like targets, so keep OFF
        flipud=0.0,

        # -----------------------------
        # Custom Albumentations (Python API only)
        # Passing this replaces the default Albumentations, while YOLO-side augments
        # like mosaic/hsv remain effective. [2](https://githubissues.com/ultralytics/ultralytics/20124)[1](https://docs.ultralytics.com/modes/train/)
        # -----------------------------
        augmentations=custom_transforms,
    )


if __name__ == "__main__":
    # This structure is also recommended to avoid multiprocessing-related errors on Windows.
    main()