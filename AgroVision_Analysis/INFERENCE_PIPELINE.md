# Inference Pipeline & Calibration Architecture

**Component:** AgroVision ML Predictor Module (`backend/ml/predictor.py`)  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. End-to-End Prediction Architecture

```
[Raw User Input (File Bytes, Path, or PIL.Image)]
                      │
                      ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: Pre-Inference Image Quality & Format Validation │
│  - Format check (JPEG, PNG, WEBP, BMP)                  │
│  - Corruption check (PIL.Image.verify)                  │
│  - Size limit (< 10 MB) & Dimensions (>= 64x64)         │
│  - Blur filter (Laplacian Variance < 50.0)              │
│  - Exposure filters (Darkness < 25, Brightness > 235)   │
└─────────────────────────────┬────────────────────────────┘
                              │ Valid Image
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Deterministic Preprocessing Transform           │
│  - Convert to 3-Channel RGB                              │
│  - Direct Resize to Square (224 x 224, Bilinear)         │
│  - Float Tensor Conversion ([0.0, 1.0])                  │
│  - ImageNet Normalization (Mean/Std)                     │
│  - Unsqueeze Batch Dimension -> (1, 3, 224, 224)         │
└─────────────────────────────┬────────────────────────────┘
                              │ Input Tensor
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: MobileNetV2 Forward Pass (CPU, No Grad)         │
│  - Computes raw logit vector: z in R^38                  │
└─────────────────────────────┬────────────────────────────┘
                              │ Raw Logits (z)
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: Post-Hoc Temperature Calibration                │
│  - Scale logits by T = 0.98057675: z_cal = z / T         │
│  - Compute probabilities: p = softmax(z_cal)             │
└─────────────────────────────┬────────────────────────────┘
                              │ Calibrated Probabilities (p)
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 5: Selective Prediction & Abstention Gating        │
│  - Max confidence: conf = max_i(p_i)                     │
│  - Decision rule:                                        │
│      IF conf < 0.30: status = "ABSTAIN"                  │
│      ELSE:           status = "PREDICTION_AVAILABLE"     │
└─────────────────────────────┬────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 6: Entity Resolution & Structured Output           │
│  - Maps index to Crop, Disease, Health Status            │
│  - Returns top-3 ranked candidate classes                │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation of Calibration & Decision Rules

1. **Forward Pass:**
   $$z = f_{\text{MobileNetV2}}(x) \quad \text{where } z \in \mathbb{R}^{38}$$

2. **Temperature Scaling:**
   $$\hat{z}_i = \frac{z_i}{T} \quad \text{with } T = 0.980576753616333$$

3. **Calibrated Probability Distribution:**
   $$\hat{p}_i = \frac{e^{\hat{z}_i}}{\sum_{j=0}^{37} e^{\hat{z}_j}} \quad \text{for } i \in [0, 37]$$

4. **Prediction Extraction:**
   $$i^* = \arg\max_{i \in [0, 37]} \hat{p}_i, \quad \text{Confidence} = \hat{p}_{i^*}$$

5. **Abstention Gating Rule:**
   $$\text{Decision Status} = \begin{cases}
   \text{"PREDICTION_AVAILABLE"}, & \text{if } \text{Confidence} \ge 0.30 \\
   \text{"ABSTAIN"}, & \text{if } \text{Confidence} < 0.30
   \end{cases}$$

---

## 3. Structured Output Contracts

### 3.1 Successful Prediction Output (`PREDICTION_AVAILABLE`)
```json
{
  "status": "PREDICTION_AVAILABLE",
  "message": "Diagnosis successfully predicted.",
  "crop": "Tomato",
  "disease": "Early blight",
  "is_healthy": false,
  "confidence": 0.9412,
  "class_index": 29,
  "canonical_class": "Tomato__Early_blight",
  "display_name": "Tomato - Early blight",
  "top_candidates": [
    {
      "class_index": 29,
      "canonical_class": "Tomato__Early_blight",
      "crop": "Tomato",
      "disease": "Early blight",
      "is_healthy": false,
      "display_name": "Tomato - Early blight",
      "confidence": 0.9412
    },
    {
      "class_index": 30,
      "canonical_class": "Tomato__Late_blight",
      "crop": "Tomato",
      "disease": "Late blight",
      "is_healthy": false,
      "display_name": "Tomato - Late blight",
      "confidence": 0.0421
    },
    {
      "class_index": 34,
      "canonical_class": "Tomato__Target_Spot",
      "crop": "Tomato",
      "disease": "Target Spot",
      "is_healthy": false,
      "display_name": "Tomato - Target Spot",
      "confidence": 0.0098
    }
  ],
  "abstain_threshold": 0.3,
  "temperature_applied": 0.980577,
  "validation_details": {
    "width": 800,
    "height": 600,
    "mean_luminance": 128.4,
    "laplacian_variance": 245.8,
    "leaf_presence_heuristic": {
      "green_pixel_ratio": 0.62,
      "yellow_brown_pixel_ratio": 0.18
    }
  }
}
```

### 3.2 Uncertain Prediction Output (`ABSTAIN`)
```json
{
  "status": "ABSTAIN",
  "message": "Confidence (24.8%) is below the diagnostic threshold (30.0%). Unable to reliably identify disease; flagged for specialist referral.",
  "crop": "Corn",
  "disease": "Cercospora leaf spot / Gray leaf spot",
  "is_healthy": false,
  "confidence": 0.248,
  "class_index": 7,
  "canonical_class": "Corn_(maize)__Cercospora_leaf_spot_Gray_leaf_spot",
  "display_name": "Corn - Cercospora leaf spot / Gray leaf spot",
  "top_candidates": [...],
  "abstain_threshold": 0.3,
  "temperature_applied": 0.980577,
  "validation_details": {...}
}
```

### 3.3 Invalid Image Output (`INVALID_IMAGE`)
```json
{
  "status": "INVALID_IMAGE",
  "reason": "IMAGE_TOO_BLURRY",
  "message": "The image is severely blurry. Please upload a clearer photograph of the leaf.",
  "details": {
    "laplacian_variance": 18.4,
    "threshold": 50.0
  }
}
```
