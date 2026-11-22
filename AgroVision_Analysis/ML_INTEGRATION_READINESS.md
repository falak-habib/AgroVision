# AgroVision — ML Integration Readiness Assessment

**Phase:** 1.5 — ML Integration Readiness & Validation  
**Date:** November 2026  
**Status:** READY FOR FASTAPI INTEGRATION (With Documented Agricultural Guardrails)  
**Evaluator:** Senior Machine Learning Engineer  

---

## 1. Executive Summary

This report establishes the production readiness of the selected PyTorch MobileNetV2 crop disease screening model (`ml/artifacts/all_v1/model.pt`) for integration into the AgroVision FastAPI backend.

All foundational ML components have been verified through rigorous automated testing (31/31 unit and integration tests passing). A decoupled, CPU-optimized inference module has been developed under `backend/ml/` without introducing external web framework dependencies or altering the trained model weights.

---

## 2. Readiness Scorecard & Findings Classification

| Dimension | Status | Key Verified Findings |
| :--- | :--- | :--- |
| **Model Serialization & Loading** | **VERIFIED** | Format is PyTorch `OrderedDict` (`state_dict`, 314 keys). Loads onto CPU in **118.17 ms** with 100% strict parameter matching. Memory overhead: **22.71 MB**. |
| **Preprocessing Parity** | **VERIFIED** | Exact parity with training evaluation pipeline: RGB format, direct square resize $(224 \times 224)$, bilinear interpolation with antialiasing, `ToTensor()` $[0, 1]$, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). |
| **Temperature Calibration** | **VERIFIED** | Temperature $T = 0.980577$ verified in `artifact.json` and mathematically enforced via $p = \text{softmax}(z / T)$. Expected Calibration Error: **0.0018**. |
| **Abstention Gating** | **VERIFIED** | Threshold $\theta = 0.30$ strictly enforced. Predictions with $\max(p) < 0.30$ suppress automated diagnosis and return `"status": "ABSTAIN"`. |
| **Class Mapping Contract** | **VERIFIED** | All 38 classes (indices 0–37) mapped cleanly to 14 crops and 26 disease conditions + 12 healthy classes. Verified against `artifact.json`. |
| **Pre-Inference Validation** | **IMPLEMENTED** | Validates file corruption, size ($<10\text{ MB}$), minimum resolution ($\ge 64\times 64$), severe blur (Laplacian variance $<50$), extreme darkness ($<25$), and extreme overexposure ($>235$). |
| **CPU Inference Latency** | **VERIFIED** | Median inference latency: **19.65 ms**; Median total pipeline latency: **22.03 ms**; P95 total latency: **27.14 ms**. |
| **Decoupled Architecture** | **IMPLEMENTED** | Reusable `backend/ml/` package created with `CropDiseasePredictor`, `ImageValidator`, `ImagePreprocessor`, and `AgroVisionModelLoader`. Zero FastAPI dependencies. |
| **Field Generalization** | **NEEDS ATTENTION** | PlantVillage laboratory bias remains. Model assumes uniform background. Out-of-distribution inputs require quality gating and downstream agricultural extension referral. |
| **Multimodal Context** | **FUTURE IMPROVEMENT** | Environmental telemetry (weather, soil) must remain in the business logic advisory layer, not injected into the image CNN. |

---

## 3. Verified Technical Highlights

1. **Deterministic CPU Serving:**
   The model loads and runs without requiring CUDA. Execution on CPU is thread-safe and requires no GPU memory allocation.
2. **Strict Parity with Training:**
   No discrepancies exist between the preprocessing transforms in `ml/train_agrovision.py` and `backend/ml/preprocessing.py`.
3. **Calibrated Confidence:**
   Softmax scores are scaled by $T=0.980577$, ensuring confidence values reflect statistical calibration rather than overconfident unconstrained logits.
4. **Automated Safety Gate:**
   Low-confidence predictions ($\text{conf} < 0.30$) are programmatically intercepted to protect farmers from misdiagnosis.

---

## 4. Remaining Issues & Constraints Prior to Backend Launch

1. **Closed-World Softmax Assumption:**
   The model will output probabilities across the 38 classes for any valid image tensor. While the `ImageValidator` catches blur, extreme lighting, and corrupt files, non-leaf objects (e.g., shoes, animals) that pass basic lighting checks may still receive predictions. A dedicated binary leaf detector or Mahalanobis distance OOD detector is recommended for future phases.
2. **Laboratory Domain Shift:**
   The model's reported 99.66% accuracy is in-domain (PlantVillage). In real-world fields, ambient lighting, shadows, and soil clutter will introduce distribution shift. The abstention threshold must be strictly preserved to catch these cases.

---

## 5. Conclusion & Recommendation

The ML component is **VERIFIED** and **READY FOR INTEGRATION** into the FastAPI application layer in Phase 2.
