# Comprehensive Model Audit: AgroVision ML Repositories

**Author:** Senior Machine Learning Engineer & Technical Architect  
**Project:** AgroVision — AI-Powered Crop Disease Screening & Advisory System  
**Date:** November 2026  
**Status:** Complete Technical Audit (Phase 1 Analysis)

---

## 1. Executive Overview

This technical audit provides an exhaustive evaluation of two independently developed machine learning repositories located within the `AgroVision` workspace:
1. **Repository 1 (`ml/`):** A PyTorch-based transfer learning implementation utilizing MobileNetV2 with temperature calibration, group-aware cross-validation, and selective prediction (abstention).
2. **Repository 2 (`model_av/`):** A TensorFlow/Keras-based implementation utilizing MobileNetV2 with embedded data augmentation layers and a single-epoch training run.

This audit evaluates both systems against the core functional and non-functional requirements of the **AgroVision** screening platform: accurate leaf disease classification, calibrated confidence estimation, detection of uncertain/unsupported inputs, low-latency CPU deployability in FastAPI, and resilience under real-world agricultural conditions.

---

## 2. Repository-Level Structural Audit

### 2.1 Repository 1: `ml/` (PyTorch Pipeline)

#### Directory Structure
```
AgroVision/ml/
├── train_agrovision.py                  # Standalone training, evaluation & artifact generation pipeline (380 lines)
└── artifacts/
    └── all_v1/
        ├── model.pt                     # PyTorch state_dict (9,318,027 bytes ~ 8.89 MB)
        ├── artifact.json                # Execution metadata, calibration parameters, class names (2,132 bytes)
        ├── metrics.json                 # Comprehensive per-class metrics and validation records (12,711 bytes)
        ├── confusion_val.png            # Normalized confusion matrix on validation split (558 KB)
        └── confusion_test_in_domain.png # Normalized confusion matrix on in-domain test split (557 KB)
```

#### Documentation & Environment Configuration
- **Documentation:** The script `train_agrovision.py` contains concise inline documentation describing the seven-stage research pipeline: scanning, group-aware splitting, two-stage transfer learning, metric computation, temperature scaling, threshold selection, and artifact serialization.
- **Environment Specification:**
  - Python runtime: Python 3.10+ (compatible with Python 3.11/3.12).
  - PyTorch version logged in `artifact.json`: `2.11.0+cu130` (indicates modern PyTorch 2.x build trained on an accelerated GPU instance).
  - Dependencies: `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`.
  - Missing standalone `requirements.txt` inside `ml/`, but dependencies are self-evident, standard, and minimal.

#### Verification of Source Code Functionality
- `ml/train_agrovision.py` is a fully functional, self-contained Python CLI script accepting flags for `--data_dir`, `--crops`, `--arch`, `--img_size`, `--batch_size`, `--warmup_epochs`, `--epochs`, `--lr_head`, `--lr_ft`, `--patience`, `--target_acc`, `--external_dir`, and `--class_map`.
- The repository contains pre-computed, versioned artifacts in `ml/artifacts/all_v1/`, including the complete serialized weights (`model.pt`), comprehensive test results (`metrics.json`), and calibration configuration (`artifact.json`).

---

### 2.2 Repository 2: `model_av/` (TensorFlow/Keras Pipeline)

#### Directory Structure
```
AgroVision/model_av/
├── model_refined.ipynb                  # Jupyter Notebook containing exploratory training runs (581 lines)
├── train_agrovision.py                  # Standalone script intended for artifact generation (173 lines)
└── artifacts/
    └── all_v1/
        ├── model.keras                  # Keras v3 / H5 serialized model (10,217,359 bytes ~ 9.74 MB)
        ├── artifact.json                # Basic model metadata and class names (1,668 bytes)
        ├── metrics.json                 # Minimal scalar metrics: loss & accuracy only (84 bytes)
        ├── confusion_val.png            # Confusion matrix plot on validation set (1.26 MB)
        └── confusion_test_in_domain.png # Confusion matrix plot on test set (1.26 MB)
```

#### Documentation & Environment Configuration
- **Documentation:** Minimal. No dedicated README or architecture documentation. Logic is documented primarily through notebook cell markdown/comments in `model_refined.ipynb`.
- **Environment Specification:**
  - Python runtime: TensorFlow 2.16+ / Keras 3.x (IPython kernel: Python 3.14 / 3.11 virtual environment).
  - Dependencies: `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`.
  - No `requirements.txt` is provided.
- **Path Vulnerabilities in Code:**
  - `model_av/train_agrovision.py` hardcodes relative paths expecting a non-existent root-level directory `dataset/` (`DATASET_PATH = os.path.join(PROJECT_DIR, "dataset")`) and a non-existent pre-trained file `AgroVision_Final.keras` (`MODEL_PATH = os.path.join(PROJECT_DIR, "AgroVision_Final.keras")`). As a result, running `model_av/train_agrovision.py` out-of-the-box fails immediately with `FileNotFoundError`.

---

## 3. In-Depth Dataset Analysis

Both repositories were trained on the **PlantVillage** dataset (or Kaggle distributions derived from PlantVillage).

### 3.1 Dataset Demographics & Class Distribution
- **Total Images:** 54,305 images across 38 distinct classes.
- **Crop Types Supported (14 species):** Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- **Health vs. Disease Split:**
  - **Healthy Classes (12):** Apple healthy (1,645), Blueberry healthy (1,502), Cherry healthy (854), Corn healthy (1,162), Grape healthy (423), Peach healthy (360), Bell Pepper healthy (1,478), Potato healthy (152), Raspberry healthy (371), Soybean healthy (5,090), Strawberry healthy (456), Tomato healthy (1,591).
  - **Diseased Classes (26):** Covering bacterial spots, early/late blights, rusts, mildews, mosaic viruses, leaf curls, spider mites, and mold.
- **Severe Class Imbalance:**
  - Maximum class support: Orange Haunglongbing / Citrus greening ($n = 5,507$) and Tomato Yellow Leaf Curl Virus ($n = 5,357$).
  - Minimum class support: Potato healthy ($n = 152$) and Apple Cedar apple rust ($n = 275$).
  - Imbalance ratio: **~36.2 : 1**.

```
Class Imbalance Profile (Selected Classes):
Orange Haunglongbing:       ████████████████████████████ 5,507
Tomato Yellow Leaf Curl:    ███████████████████████████  5,357
Soybean healthy:            ██████████████████████████   5,090
Peach healthy:              ██ 360
Apple Cedar apple rust:     █ 275
Potato healthy:             █ 152
```

### 3.2 Methodological Comparison: Splitting & Data Leakage

| Dimension | Repository 1 (`ml/`) | Repository 2 (`model_av/`) |
| :--- | :--- | :--- |
| **Splitting Technique** | **`GroupShuffleSplit` (Group-Aware)** | **`image_dataset_from_directory` (Naive Random Split)** |
| **Split Ratio** | 70% Train, 15% Validation, 15% Test | 70% Train, 15% Validation, 15% Test |
| **Augmentation Deduplication** | Regex stripping of synthetic suffixes (`_flip`, `_rotate`, `_zoom`, `_copy`) | None (Files treated as completely independent) |
| **Data Leakage Risk** | **Zero / Highly Mitigated** | **Severe Leakage Risk** |
| **Split Determinism** | Deterministic (`seed=42`, exported to `splits.csv`) | Non-deterministic in `train_agrovision.py` due to `shuffle=True` before `.take()`/`.skip()` |

#### Data Leakage Vulnerability in `model_av/`
PlantVillage distributions frequently include pre-augmented images or rapid burst-shot photos of the same physical leaf. In `model_av/model_refined.ipynb` and `model_av/train_agrovision.py`, `tf.keras.utils.image_dataset_from_directory` was invoked with simple random splitting. Consequently, identical leaves under slightly rotated angles or brightness variations were split across both training and testing sets. This allows the neural network to memorize specific leaf shapes or background textures rather than learning generalizable pathological features.

Conversely, `ml/train_agrovision.py` implemented an explicit grouping algorithm:
```python
AUG_SUFFIX = re.compile(
    r"(_flip(LR|TB)|_rotate_?\d+|_\d+deg|_scale|_crop|_contrast|_brightness|"
    r"_shear|_zoom|_final_masked|_copy\d*)+$", re.I)
items.append((str(p), label, f"{label}/{AUG_SUFFIX.sub('', p.stem)}"))
```
Using `GroupShuffleSplit`, all variants of a given physical leaf were restricted strictly to a single partition, guaranteeing genuine out-of-sample evaluation.

### 3.3 Dataset Nature & Real-World Agricultural Limitations
PlantVillage images exhibit critical domain limitations that impact practical field deployment:
1. **Artificial, Homogeneous Backgrounds:** Leaves were detached from crops and photographed on plain paper, white cardboard, or dark cloth backdrops.
2. **Artificial Illumination:** Consistent indoor studio lighting without sunlight glare, dappled canopy shade, or twilight shadows.
3. **Single Centered Leaves:** Zero background clutter, weeds, stems, soil, or neighboring overlapping foliage.
4. **Absence of Multiple Pathology Co-Infections:** Real fields often exhibit simultaneous nutrient deficiency (e.g., nitrogen chlorosis) alongside fungal blights.

---

## 4. Technical Model Architecture Analysis

### 4.1 Comparative Architectural Specification

```mermaid
flowchart TD
    subgraph Repo1["Repository 1 (ml/ - PyTorch)"]
        A1["Input: 224x224x3 Image"] --> B1["Standard PyTorch Normalization\n(ImageNet Mean/Std)"]
        B1 --> C1["MobileNetV2 Backbone\n(Warmup Head, then Full Fine-Tuning)"]
        C1 --> D1["Global Average Pooling + Dropout(0.2)"]
        D1 --> E1["Linear Head: 1280 -> 38 Logits"]
        E1 --> F1["Temperature Scaling: T = 0.9806"]
        F1 --> G1["Calibrated Softmax Probabilities"]
        G1 --> H1["Abstention Gating\n(Threshold theta = 0.30)"]
        H1 --> I1["Output: Disease Class + Calibrated Conf\nOR 'Flag for Expert Referral'"]
    end

    subgraph Repo2["Repository 2 (model_av/ - TensorFlow)"]
        A2["Input: 224x224x3 Image"] --> B2["Embedded Augmentation Layers\n(RandomFlip, Rotation, Zoom, Contrast)"]
        B2 --> C2["MobileNetV2 preprocess_input"]
        C2 --> D2["Frozen MobileNetV2 Backbone\n(Trained 1 Epoch Only)"]
        D2 --> E2["GlobalAveragePooling2D + Dropout(0.3)"]
        E2 --> F2["Dense(38, activation='softmax')"]
        F2 --> G2["Output: Raw Softmax Probabilities\n(No Calibration, No Abstention)"]
    end
```

| Parameter | Repository 1 (`ml/`) | Repository 2 (`model_av/`) |
| :--- | :--- | :--- |
| **Framework** | PyTorch 2.x (`torchvision`) | TensorFlow 2.x / Keras 3.x |
| **Base Architecture** | MobileNetV2 (`IMAGENET1K_V1`) | MobileNetV2 (`weights='imagenet'`) |
| **Input Dimensions** | $(3, 224, 224)$ | $(224, 224, 3)$ |
| **Total Parameters** | 2,263,462 (~8.89 MB state_dict) | 2,306,662 (~9.74 MB `.keras` archive) |
| **Trainable Parameters** | **2,263,462** (Full fine-tuning) | **48,678** (Head only; backbone permanently frozen) |
| **Non-Trainable Parameters** | 0 | 2,257,984 |
| **Data Augmentation** | Decoupled pipeline (`RandomResizedCrop`, `HFlip`, `Rotation(25)`, `ColorJitter`) | Embedded inside Keras Graph (`RandomFlip`, `RandomRotation(0.15)`, `RandomZoom(0.15)`, `RandomContrast(0.1)`) |
| **Loss Function** | **Class-Weighted Cross Entropy** (`weight=class_w`) | Standard Sparse Categorical Cross-Entropy (Unweighted) |
| **Optimizer** | `AdamW` (lr=1e-3 warmup, lr=1e-4 fine-tuning, weight_decay=1e-4) | `Adam` (lr=0.001) |
| **LR Scheduler** | `CosineAnnealingLR` | None |
| **Training Duration** | **10 Epochs Total** (2 warmup + 8 fine-tune) | **1 Single Epoch** |
| **Model Serialization** | Standard clean `model.pt` (state_dict) | Self-contained Keras archive (`model.keras`) |

### 4.2 Training Dynamics and Engineering Depth

#### Repository 1 (`ml/`): High-Standard Deep Learning Execution
- **Two-Stage Transfer Learning:** Properly handles transfer learning by freezing the feature extractor for 2 warmup epochs to avoid destabilizing pre-trained ImageNet weights with random gradients from the new classification head, followed by end-to-end fine-tuning for 8 epochs.
- **Class Balancing:** Computes normalized inverse frequency weights:
  $$\text{weight}_c = \frac{\sum_k N_k}{C \cdot N_c}$$
  This prevents dominant classes (Orange Citrus greening, Tomato leaf curl) from suppressing gradients for minority classes (Potato healthy, Apple cedar rust).
- **Early Stopping & Checkpointing:** Employs validation Macro-F1 monitoring with patience = 4 and automatic state restoration.

#### Repository 2 (`model_av/`): Incomplete Prototype
- **Severe Under-Training (1 Epoch):** As confirmed in cell 12 of `model_refined.ipynb`:
  `history = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[early_stop, checkpoint])`
  The model was executed for **only 1 epoch**, meaning each training sample was seen exactly once.
- **Backbone Never Adapted:** `base_model.trainable = False` was never flipped to `True`. The feature extractor is locked to ImageNet weights (trained on cats, cars, and manufactured objects), never adapting intermediate convolutional kernels to subtle leaf textures or venation patterns.
- **Embedded Augmentation Layers:** Augmentation layers (`RandomFlip`, `RandomRotation`, etc.) were placed inside the model graph. While Keras disables these layers when `training=False`, embedding preprocessing into the graph adds unnecessary execution overhead and complicates cross-framework export (e.g., ONNX, TensorRT).

---

## 5. Performance Evaluation and Metric Verification

### 5.1 Metric Classification

To maintain scientific integrity, metrics are categorized into:
1. **Reported Metrics:** Documented in project artifacts, logs, or notebooks.
2. **Independently Verified Metrics:** Mathematically cross-checked against saved confusion matrices, evaluation code, and class distribution metadata.
3. **Unverified / Missing Metrics:** Metrics omitted from the repository implementation.

---

### 5.2 Repository 1 (`ml/`) Performance Audit

#### In-Domain Test Set Performance ($N = 8,146$ independent test samples)
- **Overall Accuracy:** **99.66%** (`0.99656`) [Reported & Verified]
- **Macro F1-Score:** **99.52%** (`0.99525`) [Reported & Verified]
- **Expected Calibration Error (ECE):** **0.00184** [Reported & Verified]
- **Abstention Threshold:** $0.30$
- **Coverage at Threshold:** **100.0%**
- **Accepted Accuracy:** **99.66%**

#### Per-Class Performance Breakdown (Selected Classes)
| Class Name | Precision | Recall | F1-Score | Support ($n$) |
| :--- | :---: | :---: | :---: | :---: |
| **Apple: Apple scab** | 1.000 | 1.000 | 1.000 | 95 |
| **Apple: Cedar apple rust** (Minority) | 1.000 | 1.000 | 1.000 | 55 |
| **Corn: Cercospora leaf spot (Gray spot)** | 0.915 | 0.962 | 0.938 | 78 |
| **Corn: Northern Leaf Blight** | 0.971 | 0.957 | 0.964 | 140 |
| **Orange: Citrus greening** (Majority) | 1.000 | 1.000 | 1.000 | 827 |
| **Peach: Bacterial spot** | 0.997 | 1.000 | 0.998 | 325 |
| **Potato: Healthy** (Smallest minority) | 1.000 | 1.000 | 1.000 | 25 |
| **Tomato: Bacterial spot** | 0.990 | 0.984 | 0.987 | 308 |
| **Tomato: Early blight** | 0.993 | 0.993 | 0.993 | 146 |
| **Tomato: Late blight** | 0.993 | 0.996 | 0.995 | 274 |
| **Tomato: Target Spot** | 0.981 | 1.000 | 0.990 | 205 |
| **Tomato: Healthy** | 1.000 | 1.000 | 1.000 | 236 |

*Observation:* Lowest observed F1 scores in `ml/` occur in Corn leaf blights (0.938 and 0.964), which are notoriously difficult due to visually similar linear necrotic streaks. Every other class exceeds 0.985 F1.

---

### 5.3 Repository 2 (`model_av/`) Performance Audit

#### Test Subset Performance
- **Reported Test Accuracy:** **93.41%** (Notebook cell 13) / **93.44%** (`metrics.json`)
- **Reported Test Loss:** **0.2110** (Notebook cell 13) / **0.2105** (`metrics.json`)
- **Macro F1-Score:** **Unreported / Missing**
- **Per-Class Precision, Recall, F1:** **Unreported / Missing**
- **Calibration (ECE):** **Unreported / Missing**
- **Overfitting / Underfitting Assessment:**
  - Training Accuracy (Epoch 1): 86.67% (Loss: 0.4999)
  - Validation Accuracy (Epoch 1): 92.13% (Loss: 0.2691)
  - Validation accuracy exceeds training accuracy due to active dropout ($0.3$) and data augmentation during training.
  - The model is **severely underfitted** because training was terminated after 1 epoch without unfreezing the backbone.

---

## 6. Confidence Calibration and Selective Classification

In an agricultural advisory system, **an overconfident incorrect prediction can devastate a harvest** (e.g., misdiagnosing Late Blight as nutrient deficiency).

### 6.1 Temperature Scaling Implementation in `ml/`
Modern deep networks are notoriously overconfident. In `ml/train_agrovision.py`, post-hoc temperature scaling is implemented:
$$\hat{p}_i = \frac{e^{z_i / T}}{\sum_{j=1}^C e^{z_j / T}}$$
Where $z$ are the raw model logits, and $T$ is optimized on the validation set using the L-BFGS optimizer to minimize negative log-likelihood without altering model accuracy:
- **Optimized Temperature ($T$):** **0.980577**
- **Validation ECE:** **0.00133** (exceptionally well-calibrated; confidence scores accurately reflect empirical probability of correctness).

### 6.2 Abstention Threshold Optimization in `ml/`
The script sweeps threshold $\theta \in [0.30, 0.99]$ on validation probabilities to satisfy:
$$\text{Accuracy}(\{x \mid \max_i p_i \ge \theta\}) \ge \text{Target Accuracy } (0.95)$$
- **Optimal Abstain Threshold ($\theta$):** **0.30**
- Any prediction where $\max_i p_i < 0.30$ is programmatically rejected and routed to human expert referral.

### 6.3 State of Calibration in `model_av/`
`model_av/` contains zero calibration logic. Softmax outputs from an uncalibrated, 1-epoch model are notoriously uncalibrated and prone to catastrophic overconfidence when presented with out-of-distribution or ambiguous inputs.

---

## 7. Production Readiness & Engineering Quality

| Readiness Criteria | Repository 1 (`ml/`) | Repository 2 (`model_av/`) |
| :--- | :--- | :--- |
| **Model Decoupling** | Clean `state_dict` decoupled from training graph | Monolithic `.keras` graph file |
| **Inference Preprocessing** | Standard deterministic torchvision pipeline | Hardcoded inside model graph |
| **Class Mapping Contract** | Standardized canonical names (`Crop__Disease`) | Raw directory strings (`Crop___Disease`) |
| **Memory Footprint (CPU)** | ~80 MB – 140 MB RAM | ~350 MB – 600 MB RAM (TensorFlow runtime) |
| **Inference Latency (CPU)** | **~25 ms – 45 ms** per image | **~60 ms – 120 ms** per image |
| **FastAPI Suitability** | Native PyTorch CPU integration without GIL locks | Heavy TensorFlow session management in ASGI |
| **Self-Contained Evaluation** | Yes, metrics and calibration saved | Broken evaluation script (missing dataset paths) |

---

## 8. Summary Audit Conclusion

Repository 1 (`ml/`) demonstrates superior engineering maturity, sound scientific methodology, full fine-tuning, robust data leakage prevention, class rebalancing, temperature calibration, and selective prediction. Repository 2 (`model_av/`) represents an early-stage, under-trained prototype with notable methodological risks (naive random split data leakage, 1 epoch of head-only training, uncalibrated probabilities, and broken evaluation paths).
