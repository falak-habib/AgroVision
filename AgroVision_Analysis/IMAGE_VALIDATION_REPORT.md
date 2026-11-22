# Image Quality & Input Validation Report

**Component:** Image Validation Subsystem (`backend/ml/image_validator.py`)  
**Status:** IMPLEMENTED & BENCHMARKED  

---

## 1. Overview & Operational Need

The AgroVision MobileNetV2 classifier is a closed-world model. Without pre-inference input validation, sending corrupted files, microscopic thumbnails, extreme glare, out-of-focus blur, or pitch-black photos forces the model to emit arbitrary high-confidence diagnoses.

The `ImageValidator` acts as an active front-line gate, evaluating images *before* they consume CPU resources in the neural network.

---

## 2. Validation Checks & Threshold Rationale

| Check | Method & Metric | Threshold | Action on Violation |
| :--- | :--- | :--- | :--- |
| **File Integrity** | `PIL.Image.open()` + `verify()` | Valid header, non-corrupted stream | Reject (`CORRUPT_OR_UNREADABLE_IMAGE`) |
| **File Size** | Byte length | Max $10\text{ MB}$ ($10,485,760$ bytes) | Reject (`FILE_TOO_LARGE`) |
| **Resolution** | Min(Width, Height) | Min $64 \times 64$ pixels | Reject (`IMAGE_RESOLUTION_TOO_LOW`) |
| **Format** | Image format string | JPEG, PNG, WEBP, BMP | Reject (`UNSUPPORTED_IMAGE_FORMAT`) |
| **Motion/Focus Blur** | Discrete Laplacian Variance $\text{Var}(\nabla^2 I_{\text{gray}})$ | Variance $< 50.0$ | Reject (`IMAGE_TOO_BLURRY`) |
| **Extreme Darkness** | Grayscale mean intensity $\bar{I}$ | $\bar{I} < 25.0$ (scale $0-255$) | Reject (`IMAGE_TOO_DARK`) |
| **Overexposure / Glare**| Grayscale mean intensity $\bar{I}$ or Saturated pixel fraction | $\bar{I} > 235.0$ or $>85\%$ pixels at $\ge 250$ | Reject (`IMAGE_TOO_BRIGHT`) |
| **Color Space** | Pillow mode conversion | Converts to RGB | Normalizes RGBA / L / P modes |

---

## 3. The Leaf Presence Dilemma: Greenness vs. Disease Necrosis

A naive instinct in agricultural computer vision is to reject any image lacking high green pixel coverage (e.g., using HSV Hue or Excess Green Index $ExG = 2G - R - B$).

### Critical Research Finding:
> **A strict green-pixel filter is catastrophically counterproductive for plant pathology.**

When leaves develop severe disease:
- **Apple Black Rot & Grape Black Rot:** Leaf tissue turns dark charcoal and black.
- **Potato & Tomato Late Blight:** Leaves become water-soaked, turning dark brown and collapsed.
- **Strawberry Leaf Scorch & Apple Cedar Rust:** Foliage turns bright red, purple, orange, and necrotic brown.
- **Tomato Yellow Leaf Curl Virus & Chlorosis:** Foliage loses chlorophyll completely, turning pale yellow or white.

If the validation module rejected images with low green pixel ratios, **it would reject the most critically diseased crops requiring urgent intervention!**

### Adopted Design:
The `ImageValidator` calculates $ExG$ and yellow-brown necrotic pixel ratios purely for **diagnostic metadata logging**, but **never rejects an image based solely on color hue**.

---

## 4. Limitations of Heuristic Image Processing

While Laplacian variance and luminance thresholding catch poor-quality photography, classical image processing cannot reliably differentiate:
- A sharp photograph of a healthy leaf from a sharp photograph of a green towel, green car, or plastic cup.
- A leaf photographed from 10 meters away (canopy scale) vs. a macro-photograph of a single lesion.

---

## 5. Architectural Recommendation for Phase 3: Dedicated Binary Leaf Detector

To achieve robust out-of-distribution rejection in production:
1. **Lightweight Binary Leaf Classifier:** Train a tiny MobileNetV3-Small or ShuffleNetV2 ($< 1.5\text{ MB}$) on a binary task: `Leaf` vs. `Non-Leaf` (trained against ImageNet non-plant distractors).
2. **YOLOv8-Nano Foliage Bounding Box Detector:** Detect individual leaf blades in complex field backgrounds and crop the primary diseased leaf before passing it to MobileNetV2.
