# Primary Model Selection & Architectural Recommendation

**Author:** Senior Machine Learning Engineer & Technical Architect  
**Project:** AgroVision  
**Date:** November 2026  
**Status:** Approved Technical Recommendation  

---

## 1. Definitive Recommendation

After a comprehensive technical audit of both repositories, **Repository 1 (`ml/` - PyTorch MobileNetV2)** is conclusively recommended as the primary machine learning model for integration into the **AgroVision** screening system.

Repository 2 (`model_av/` - TensorFlow/Keras) is **rejected** for production deployment.

```
+-----------------------------------------------------------------------------------------+
|                                RECOMMENDED PRIMARY MODEL                                |
|                                                                                         |
|  Repository:        AgroVision/ml/                                                      |
|  Architecture:      MobileNetV2 (Two-Stage Transfer Learning: Warmup + Fine-Tuning)     |
|  Weights Path:      AgroVision/ml/artifacts/all_v1/model.pt                             |
|  Configuration:     AgroVision/ml/artifacts/all_v1/artifact.json                        |
|  Metrics Record:    AgroVision/ml/artifacts/all_v1/metrics.json                         |
|  Framework:         PyTorch (TorchScript / Standard Eager CPU compatible)               |
|  Accuracy:          99.66% In-Domain Test Accuracy (Macro F1: 99.52%)                   |
|  Calibration:       Temperature-Scaled (T = 0.9806, ECE = 0.0018)                       |
|  Abstention:        Active Confidence Gate (Threshold theta = 0.30)                     |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Technical Justification for Selecting Repository 1 (`ml/`)

The selection of Repository 1 is based on sound machine learning principles, robust empirical evidence, and engineering suitability:

### 2.1 Complete and Rigorous Transfer Learning
- In agricultural leaf screening, generic pre-trained ImageNet filters (detecting edges, wheels, fur) must be specialized to detect subtle botanical signs (chlorosis, necrotic halos, fungal pustules, and leaf venation).
- `ml/` executed a **two-phase training protocol**:
  1. *Stage 1 (Warmup):* Frozen feature extractor, optimizing only the linear head for 2 epochs using AdamW ($10^{-3}$) to initialize head weights smoothly.
  2. *Stage 2 (Fine-Tuning):* Full model end-to-end training for 8 epochs at a reduced learning rate ($10^{-4}$) with cosine annealing scheduling.
- In contrast, `model_av/` locked the backbone permanently and ran for **only 1 epoch**, leaving the feature extractor completely unadapted to botanical pathology.

### 2.2 Mitigation of Data Leakage (Group-Aware Splitting)
- PlantVillage datasets frequently contain near-duplicate frames, augmented copies, and sequential camera bursts of the same physical leaf.
- `ml/` implemented a regex-based parser (`AUG_SUFFIX`) to group augmented images under their parent leaf ID and partitioned the data using `GroupShuffleSplit`.
- This ensures zero overlap of leaf identities across training, validation, and testing sets. Its reported 99.66% test accuracy represents genuine out-of-sample performance on independent leaves.

### 2.3 Class Imbalance Compensation
- The dataset exhibits a 36:1 class imbalance (e.g., Orange Citrus Greening has 5,507 images, while Potato Healthy has only 152 images).
- `ml/` explicitly corrected for this imbalance by computing inverse-frequency loss weights and passing them into `F.cross_entropy(weight=class_w)`.
- As a result, minority classes achieved remarkable performance:
  - Potato Healthy (152 total samples, 25 test samples): **100% Precision, 100% Recall, 1.000 F1**.
  - Apple Cedar Apple Rust (275 total samples, 55 test samples): **100% Precision, 100% Recall, 1.000 F1**.
- `model_av/` used unweighted loss, allowing majority classes to dominate model gradients.

### 2.4 Calibrated Confidence & Selective Classification
- AgroVision's fundamental clinical requirement is: *"Uncertain predictions and unsupported inputs should be flagged rather than presented as reliable diagnoses."*
- `ml/` is the only model that implements:
  - **Temperature Scaling ($T = 0.980577$):** Optimizes the logit temperature via L-BFGS on validation data, achieving an ultra-low Expected Calibration Error (ECE = 0.0018 on test).
  - **Abstention Gating ($\theta = 0.30$):** Samples with maximum probability $< 0.30$ are automatically flagged for agricultural extension specialist review.

### 2.5 Clean, Decoupled Production Artifacts
- The model weights are serialized as a standard PyTorch `state_dict` (`model.pt`, 8.89 MB).
- Configuration metadata is cleanly preserved in `artifact.json`, documenting input dimensions, normalization tensors (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), temperature, and threshold.
- No training code, data loaders, or augmentation layers are coupled to the model artifact, enabling trivial loading inside FastAPI.

---

## 3. Critical Assessment of Rejected Model (`model_av/`)

Repository 2 (`model_av/`) cannot be approved for production due to significant technical vulnerabilities:

1. **Severely Incomplete Training (1 Epoch):** As confirmed by cell 12 of `model_refined.ipynb`:
   ```python
   history = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[early_stop, checkpoint])
   ```
   Training for 1 single epoch on a frozen backbone means the model is severely underfitted.
2. **Backbone Feature Disconnection:** The MobileNetV2 backbone was never fine-tuned (`base_model.trainable = False`). The classifier relies purely on generic ImageNet features.
3. **Data Leakage Risk:** Used naive random splitting via `image_dataset_from_directory()`. Augmented copies of identical leaves were leaked across train and test sets, inflating test metrics.
4. **Uncalibrated Predictions:** Raw softmax output without temperature scaling or abstention thresholds.
5. **Coupled Model Graph:** Data augmentation layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`) are embedded directly inside the serialized Keras graph, adding inference overhead and potential serialization friction.
6. **Broken Artifact Script:** `model_av/train_agrovision.py` hardcodes paths to non-existent directories (`dataset/` and `AgroVision_Final.keras`), rendering the script unrunnable out-of-the-box.

---

## 4. Reusable Elements from Repository 2 (`model_av/`)

Although the trained model in `model_av/` is rejected, certain components and assets can be repurposed:
1. **Exploratory Visualizations:** The confusion matrix plotting routines utilizing `ConfusionMatrixDisplay` in `model_av/train_agrovision.py` provide clear, high-resolution rendering configurations that can be adopted for automated reporting.
2. **Class Name Directory Mapping:** The raw directory names preserved in `model_av/artifacts/all_v1/artifact.json` provide a direct cross-reference against original Kaggle/PlantVillage directory archives.

---

## 5. Remaining Limitations of the Selected Model

While `ml/` is technically superior, the following inherent limitations remain:
1. **Controlled-Environment Bias:** The model was trained on PlantVillage (detached leaves on plain grey/black sheets under studio lighting). It has not yet been fine-tuned on natural field photography with complex foliage/soil clutter.
2. **Closed-World Softmax:** The model assumes any uploaded image belongs to one of the 38 classes. Uploading an out-of-distribution image (e.g., a tractor, a human hand, or an unsupported crop like Rice or Wheat) will yield arbitrary probabilities across the 38 classes.
3. **Single Pathology Assumption:** The network outputs a single categorical label and cannot diagnose co-occurring diseases on the same leaf.

---

## 6. Prerequisite Improvements Prior to Deployment

Before exposing the model via the public AgroVision FastAPI service, the following technical safeguards must be implemented:

1. **Quality Gating / Leaf Verification:** Implement a lightweight leaf-validation or sharpness/greenness filter to intercept non-leaf or completely blurred uploads before model inference.
2. **Abstention Gating Enforcement:** The FastAPI inference handler must strictly enforce the `0.30` confidence threshold, returning an explicit `"status": "ABSTAIN"` response when confidence is insufficient.
3. **Contextual Enrichment (Non-Multimodal):** Environmental parameters (temperature, humidity, rainfall) must be integrated at the **business logic / advisory layer** as agronomic rules, and NOT injected into the CNN image feature vector (see `MODEL_RISKS.md`).
4. **Field Dataset Fine-Tuning Roadmap:** Future model retraining should incorporate field datasets (such as PlantDoc) using the existing `--external_dir` facility in `ml/train_agrovision.py`.
