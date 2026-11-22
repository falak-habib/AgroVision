# FastAPI Integration Requirements & Technical Specification

**Author:** Senior Machine Learning Engineer & Technical Architect  
**Project:** AgroVision  
**Date:** November 2026  
**Status:** Architectural Specification for Phase 2 Implementation  

---

## 1. Scope & Objective

This document defines the exact technical requirements, mathematical transformations, data structures, and code contracts necessary to integrate the selected **Repository 1 (`ml/` - PyTorch MobileNetV2)** model into a high-performance **FastAPI** backend service.

---

## 2. Model Artifact Specifications & Exact Paths

All components required for production serving are located in the `ml/artifacts/all_v1/` directory:

| Component | Exact Filesystem Path | Size | Description |
| :--- | :--- | :--- | :--- |
| **Model Weights** | `AgroVision/ml/artifacts/all_v1/model.pt` | 9,318,027 bytes (~8.89 MB) | Serialized PyTorch `state_dict` of fine-tuned MobileNetV2. |
| **Model Metadata** | `AgroVision/ml/artifacts/all_v1/artifact.json` | 2,132 bytes | Configuration: classes, normalization constants, $T$, $\theta$. |
| **Benchmark Metrics**| `AgroVision/ml/artifacts/all_v1/metrics.json` | 12,711 bytes | Reference baseline metrics for automated regression testing. |

---

## 3. Class Mapping & Entity Resolution (38 Classes)

The model maps a 0-indexed integer output $i \in [0, 37]$ to the following canonical classes, which decompose cleanly into botanical crop and condition entities:

| Index | Canonical Class String | Crop Entity | Disease / Condition Entity | Is Healthy? |
| :---: | :--- | :--- | :--- | :---: |
| **0** | `Apple__Apple_scab` | Apple | Apple Scab (*Venturia inaequalis*) | No |
| **1** | `Apple__Black_rot` | Apple | Black Rot (*Diplodia seriata*) | No |
| **2** | `Apple__Cedar_apple_rust` | Apple | Cedar Apple Rust (*Gymnosporangium juniperi-virginianae*) | No |
| **3** | `Apple__healthy` | Apple | Healthy | **Yes** |
| **4** | `Blueberry__healthy` | Blueberry | Healthy | **Yes** |
| **5** | `Cherry_(including_sour)__Powdery_mildew` | Cherry | Powdery Mildew (*Podosphaera clandestina*) | No |
| **6** | `Cherry_(including_sour)__healthy` | Cherry | Healthy | **Yes** |
| **7** | `Corn_(maize)__Cercospora_leaf_spot_Gray_leaf_spot` | Corn | Cercospora Leaf Spot / Gray Leaf Spot | No |
| **8** | `Corn_(maize)__Common_rust_` | Corn | Common Rust (*Puccinia sorghi*) | No |
| **9** | `Corn_(maize)__Northern_Leaf_Blight` | Corn | Northern Leaf Blight (*Exserohilum turcicum*) | No |
| **10** | `Corn_(maize)__healthy` | Corn | Healthy | **Yes** |
| **11** | `Grape__Black_rot` | Grape | Black Rot (*Guignardia bidwellii*) | No |
| **12** | `Grape__Esca_(Black_Measles)` | Grape | Esca / Black Measles (*Phaeoacremonium*) | No |
| **13** | `Grape__Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape | Leaf Blight (*Pseudocercospora vitis*) | No |
| **14** | `Grape__healthy` | Grape | Healthy | **Yes** |
| **15** | `Orange__Haunglongbing_(Citrus_greening)` | Orange | Huanglongbing / Citrus Greening (*Candidatus Liberibacter*) | No |
| **16** | `Peach__Bacterial_spot` | Peach | Bacterial Spot (*Xanthomonas arboricola*) | No |
| **17** | `Peach__healthy` | Peach | Healthy | **Yes** |
| **18** | `Pepper,_bell__Bacterial_spot` | Bell Pepper | Bacterial Spot (*Xanthomonas campestris*) | No |
| **19** | `Pepper,_bell__healthy` | Bell Pepper | Healthy | **Yes** |
| **20** | `Potato__Early_blight` | Potato | Early Blight (*Alternaria solani*) | No |
| **21** | `Potato__Late_blight` | Potato | Late Blight (*Phytophthora infestans*) | No |
| **22** | `Potato__healthy` | Potato | Healthy | **Yes** |
| **23** | `Raspberry__healthy` | Raspberry | Healthy | **Yes** |
| **24** | `Soybean__healthy` | Soybean | Healthy | **Yes** |
| **25** | `Squash__Powdery_mildew` | Squash | Powdery Mildew (*Podosphaera xanthii*) | No |
| **26** | `Strawberry__Leaf_scorch` | Strawberry | Leaf Scorch (*Diplocarpon earlianum*) | No |
| **27** | `Strawberry__healthy` | Strawberry | Healthy | **Yes** |
| **28** | `Tomato__Bacterial_spot` | Tomato | Bacterial Spot (*Xanthomonas campestris*) | No |
| **29** | `Tomato__Early_blight` | Tomato | Early Blight (*Alternaria solani*) | No |
| **30** | `Tomato__Late_blight` | Tomato | Late Blight (*Phytophthora infestans*) | No |
| **31** | `Tomato__Leaf_Mold` | Tomato | Leaf Mold (*Passalora fulva*) | No |
| **32** | `Tomato__Septoria_leaf_spot` | Tomato | Septoria Leaf Spot (*Septoria lycopersici*) | No |
| **33** | `Tomato__Spider_mites_Two-spotted_spider_mite` | Tomato | Two-Spotted Spider Mite (*Tetranychus urticae*) | No |
| **34** | `Tomato__Target_Spot` | Tomato | Target Spot (*Corynespora cassiicola*) | No |
| **35** | `Tomato__Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Tomato Yellow Leaf Curl Virus (TYLCV) | No |
| **36** | `Tomato__Tomato_mosaic_virus` | Tomato | Tomato Mosaic Virus (ToMV) | No |
| **37** | `Tomato__healthy` | Tomato | Healthy | **Yes** |

---

## 4. Input Preprocessing & Normalization Pipeline

To avoid train-serving skew, inference preprocessing must match the exact evaluation transform defined in `ml/train_agrovision.py`:

```
Uploaded Image Bytes
        │
        ▼
   [PIL.Image.open(bytes).convert("RGB")]
        │
        ▼
   [Direct Resize to Square: 224 x 224]  <-- transforms.Resize((224, 224))
        │
        ▼
   [Convert to Float Tensor: [0.0, 1.0]] <-- transforms.ToTensor()
        │
        ▼
   [ImageNet Normalization]              <-- transforms.Normalize(mean, std)
        │                                    mean = [0.485, 0.456, 0.406]
        ▼                                    std  = [0.229, 0.224, 0.225]
   [Batch Dimension: (1, 3, 224, 224)]
```

---

## 5. Mathematical Inference & Calibration Formulation

The inference service must execute the following mathematical operations:

1. **Forward Pass:** Compute raw unbounded logit vector $z \in \mathbb{R}^{38}$:
   $$z = f_{\theta}(x)$$

2. **Temperature Calibration:** Scale logits by the learned temperature parameter $T = 0.98057675$:
   $$\hat{z} = \frac{z}{T}$$

3. **Softmax Probability Computation:**
   $$\hat{p}_i = \frac{e^{\hat{z}_i}}{\sum_{j=0}^{37} e^{\hat{z}_j}} \quad \text{for } i \in [0, 37]$$

4. **Prediction Extraction:**
   $$i^* = \arg\max_{i} \hat{p}_i, \quad \text{Confidence} = \hat{p}_{i^*}$$

5. **Abstention Gating:**
   $$\text{Decision} = \begin{cases} 
   \text{ACCEPT}, & \text{if } \text{Confidence} \ge \theta_{\text{abstain}} \; (0.30) \\
   \text{ABSTAIN}, & \text{if } \text{Confidence} < \theta_{\text{abstain}} \; (0.30)
   \end{cases}$$

If $\text{Decision} = \text{ABSTAIN}$, the system returns a status indicating uncertainty, flagging the sample for human agricultural extension review.

---

## 6. Runtime & Environment Dependencies

The production environment should run on standard CPU instances without requiring CUDA GPU overhead.

### 6.1 Python Dependencies (`requirements.txt`)
```txt
fastapi>=0.110.0,<1.0.0
uvicorn[standard]>=0.28.0,<1.0.0
pydantic>=2.6.0,<3.0.0
python-multipart>=0.0.9
torch>=2.1.0,<2.3.0
torchvision>=0.16.0,<0.18.0
pillow>=10.2.0,<11.0.0
numpy>=1.26.0,<2.0.0
```

### 6.2 Computational & Memory Profiling (CPU Deployment)
- **Container Memory Baseline:** ~180 MB RAM.
- **Model in Memory:** ~40 MB RAM.
- **Peak RAM Under 50 Concurrent Requests:** **~450 MB – 650 MB**.
- **CPU Single-Inference Latency:** **~25 ms – 45 ms** (on 2 vCPU Intel Xeon or AMD EPYC).
- **Throughput:** ~22 – 35 requests/second per container replica.

---

## 7. FastAPI Service Implementation Architecture (Phase 2 Blueprint)

The following architectural blueprint defines the `AgroVisionModelService` class and FastAPI schemas to be implemented in Phase 2:

```python
# Blueprint for Phase 2: app/services/model_service.py
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class AgroVisionModelService:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        with open(artifact_dir / "artifact.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.classes = self.metadata["classes"]
        self.img_size = self.metadata["input_size"]
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.abstain_threshold = float(self.metadata.get("abstain_threshold", 0.30))

        # Build architecture matching training definition
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.classes))
        
        # Load weights on CPU
        state_dict = torch.load(artifact_dir / "model.pt", map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Deterministic preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.metadata["normalize_mean"],
                std=self.metadata["normalize_std"]
            )
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        # Preprocessing
        img_rgb = image.convert("RGB")
        tensor = self.transform(img_rgb).unsqueeze(0)  # Shape: (1, 3, 224, 224)

        # Forward pass & temperature scaling
        logits = self.model(tensor)
        calibrated_logits = logits / self.temperature
        probs = torch.softmax(calibrated_logits, dim=1).squeeze(0)

        confidence, pred_idx = torch.max(probs, dim=0)
        confidence = float(confidence.item())
        pred_idx = int(pred_idx.item())

        raw_label = self.classes[pred_idx]
        crop, condition = raw_label.split("__", 1)
        is_healthy = condition.lower() == "healthy"
        needs_referral = confidence < self.abstain_threshold

        return {
            "predicted_class": raw_label,
            "crop": crop.replace("_", " "),
            "condition": condition.replace("_", " "),
            "is_healthy": is_healthy,
            "confidence": round(confidence, 4),
            "abstain_threshold": self.abstain_threshold,
            "status": "ABSTAIN" if needs_referral else "ACCEPTED",
            "needs_referral": needs_referral
        }
```

---

## 8. API Request/Response Data Contract

### 8.1 Pydantic Response Schema Definition
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class EnvironmentalContext(BaseModel):
    temperature_celsius: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    rainfall_mm: Optional[float] = None
    soil_condition: Optional[str] = None
    location_region: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_class: str
    crop: str
    condition: str
    is_healthy: bool
    confidence: float
    status: str = Field(description="'ACCEPTED' or 'ABSTAIN'")
    needs_referral: bool
    agricultural_recommendations: List[str]
    context_validation_note: Optional[str] = None
```

### 8.2 Sample JSON Response (High Confidence Diagnosis)
```json
{
  "predicted_class": "Tomato__Early_blight",
  "crop": "Tomato",
  "condition": "Early blight",
  "is_healthy": false,
  "confidence": 0.9842,
  "status": "ACCEPTED",
  "needs_referral": false,
  "agricultural_recommendations": [
    "Prune lower infected foliage to prevent fungal spore splash.",
    "Apply copper-based or chlorothalonil fungicide following manufacturer guidelines.",
    "Ensure drip irrigation rather than overhead watering to reduce leaf wetness."
  ],
  "context_validation_note": "Observed temperature (24°C) and high humidity (82%) support Early Blight sporulation."
}
```

### 8.3 Sample JSON Response (Uncertain / Abstain Referral)
```json
{
  "predicted_class": "Corn_(maize)__Northern_Leaf_Blight",
  "crop": "Corn (maize)",
  "condition": "Northern Leaf Blight",
  "is_healthy": false,
  "confidence": 0.2415,
  "status": "ABSTAIN",
  "needs_referral": true,
  "agricultural_recommendations": [
    "Confidence score (24.15%) is below the diagnostic threshold (30.0%).",
    "Automated diagnosis withheld to prevent misdiagnosis.",
    "Please capture a clearer, well-lit photo of the affected leaf or consult your local agricultural extension service."
  ],
  "context_validation_note": "Prediction flagged for manual verification."
}
```
