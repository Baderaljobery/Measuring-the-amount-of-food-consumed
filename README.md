Precision Food Detector 

Estimate remaining/consumed food from before/after photos using classic computer vision (no deep learning).

Features

Works offline — no YOLO/DL models.

Precise in-plate segmentation via Gaussian Mixture Model (GMM) on RGB/HSV/LAB + spatial cues.

Morphological cleanup and built-in visualizations.

Requirements
python 3.8+
pip install opencv-python numpy matplotlib scikit-learn

Quick Start

Put the code in: precision_food_detector.py

from precision_food_detector import precision_food_analysis

consumed = precision_food_analysis("examples/before.jpg", "examples/after.jpg")
print(f"Consumed: {consumed:.1f}%")


Single-image check:

from precision_food_detector import test_single_image
test_single_image("examples/before.jpg")


Tip: unify image sizes (resize) if before/after dimensions differ.

How It Works (brief)

Preprocessing: Bilateral filter + CLAHE(LAB) for contrast.

Plate mask: multi-scale Canny → largest contour → slight erode to avoid rims.

GMM segmentation: cluster in-plate pixels using color (RGB/HSV/LAB) + position (x/w, y/h).

Food decision rules: saturation/value range, color variance, minimum cluster size.

Refinement: morphological open/close + small-component removal.

Metrics: remaining% = (after/ before) × 100; consumed% = 100 − remaining.

Tuning

analyze_cluster_for_food(...): adjust Saturation/Value thresholds and min cluster size to your data.

Morphology kernels (3,3) / (7,7): increase if masks look fragmented.

debug_mode=True for concise logs.

Performance Tips

Downscale to ~800 px width before GMM.

Optionally sample in-plate pixels (e.g., every 2–3 px) for speed.

Keep lighting, distance, and angle consistent.

Troubleshooting

Cannot load image → check paths/filenames.

Weak masks → improve lighting, use light plate, relax/tighten thresholds, enlarge morphology kernels.

Size mismatch → resize before/after to the same dimensions.
