# Real-World Generalization & External Dataset Roadmap

**Topic:** Field Robustness, Domain Shift, and PlantDoc Evaluation  
**Status:** ARCHITECTURAL ROADMAP & PROTOCOL DEFINITION  

---

## 1. The Core Limitation: The PlantVillage Domain Gap

The current model was trained and evaluated on **PlantVillage** ($N = 54,305$ images across 38 classes). While in-domain test accuracy reached 99.66%, this high accuracy is partially aided by controlled laboratory conditions:
- Detached single leaves placed flat on uniform gray, black, or sheet backgrounds.
- High-contrast illumination without direct sunlight glare or tree canopy shadows.
- Absence of surrounding soil, mulch, weeds, insect pests, or neighboring foliage.

In real-world smallholder farm settings, leaves are attached to living crops surrounded by complex agricultural clutter. Published literature shows unadapted PlantVillage models experience a 30%–60% accuracy drop when evaluated directly on field imagery.

---

## 2. Terminology: Distinguishing Evaluation & Training Modalities

To avoid methodological confusion, the development roadmap formally defines:

1. **In-Domain Evaluation:**
   Testing the model on held-out samples originating from the exact same dataset distribution (e.g., the 8,146 held-out PlantVillage test samples).
2. **External Validation (Out-of-Domain Zero-Shot Testing):**
   Evaluating the frozen model directly on an independently collected field dataset (e.g., PlantDoc) without any parameter updates. Measures true domain transfer robustness.
3. **Fine-Tuning:**
   Taking the pre-trained weights and continuing backpropagation on a combination of laboratory and field images using a reduced learning rate ($10^{-5}$).
4. **Domain Adaptation:**
   Techniques (e.g., adversarial domain confusion, CORAL, or test-time normalization) designed to align the feature distributions of laboratory source data and farm target data without requiring full field label annotation.

---

## 3. External Field Dataset: PlantDoc Integration Analysis

**PlantDoc** is a publicly available benchmark dataset (Singh et al., 2020) containing 2,598 agricultural field images across 27 classes captured in real farms.

### 3.1 Class Compatibility & Mapping Challenges

Not all classes in PlantDoc correspond to PlantVillage classes:

| Crop | PlantDoc Class | Equivalent PlantVillage Class in `ml/` | Compatibility Status |
| :--- | :--- | :--- | :--- |
| **Apple** | Apple Scab Leaf | `Apple__Apple_scab` | **Direct Match** |
| **Apple** | Apple leaf | `Apple__healthy` | **Direct Match** |
| **Apple** | Apple Black rot | `Apple__Black_rot` | **Direct Match** |
| **Apple** | Cedar apple rust leaf | `Apple__Cedar_apple_rust` | **Direct Match** |
| **Corn** | Corn leaf blight | `Corn_(maize)__Northern_Leaf_Blight` | Partial / Needs Mapping |
| **Corn** | Corn Gray leaf spot | `Corn_(maize)__Cercospora_leaf_spot_Gray_leaf_spot` | **Direct Match** |
| **Corn** | Corn rust leaf | `Corn_(maize)__Common_rust_` | **Direct Match** |
| **Grape** | Grape leaf | `Grape__healthy` | **Direct Match** |
| **Grape** | Grape Black rot | `Grape__Black_rot` | **Direct Match** |
| **Potato** | Potato leaf early blight | `Potato__Early_blight` | **Direct Match** |
| **Potato** | Potato leaf late blight | `Potato__Late_blight` | **Direct Match** |
| **Tomato** | Tomato Early blight leaf | `Tomato__Early_blight` | **Direct Match** |
| **Tomato** | Tomato Late blight leaf | `Tomato__Late_blight` | **Direct Match** |
| **Tomato** | Tomato Septoria leaf spot | `Tomato__Septoria_leaf_spot` | **Direct Match** |
| **Tomato** | Tomato leaf yellow virus | `Tomato__Tomato_Yellow_Leaf_Curl_Virus` | **Direct Match** |
| **Tomato** | Tomato leaf bacterial spot | `Tomato__Bacterial_spot` | **Direct Match** |
| **Tomato** | Tomato leaf mold | `Tomato__Leaf_Mold` | **Direct Match** |
| **Tomato** | Tomato leaf | `Tomato__healthy` | **Direct Match** |
| **Bell Pepper** | Bell_pepper leaf | `Pepper,_bell__healthy` | **Direct Match** |
| **Bell Pepper** | Bell_pepper leaf spot | `Pepper,_bell__Bacterial_spot` | **Direct Match** |

**Excluded / Incompatible Classes:**
- PlantDoc contains classes not supported in PlantVillage (e.g., Cherry leaf blight is different from cherry powdery mildew).
- Multiple diseases can appear on a single leaf in PlantDoc, challenging single-label evaluation.

---

## 4. Proposed External Evaluation Protocol (Phase 3 Execution)

When external evaluation is initiated:
1. **Never Mix Datasets:** Do not blend external field images into the existing test set; keep test sets strictly partitioned to monitor domain gap metrics independently.
2. **Leverage Existing Pipeline Infrastructure:**
   Use the pre-built `--external_dir` and `--class_map` CLI flags already implemented in `ml/train_agrovision.py`:
   ```bash
   python ml/train_agrovision.py --data_dir PlantVillage \
       --external_dir PlantDoc_Test \
       --class_map plantdoc_mapping.csv
   ```
3. **Track Gap Metrics:**
   Measure Macro-F1 degradation $\Delta \text{F1} = \text{F1}_{\text{in-domain}} - \text{F1}_{\text{external}}$.

---

## 5. Phased Improvement Roadmap

```
PHASE 1.5 (Current):
  - In-domain MobileNetV2 verified on CPU
  - Pre-inference quality gating (blur, exposure)
  - Abstention threshold (0.30) activated
         │
         ▼
PHASE 2 (Upcoming):
  - FastAPI integration
  - Reusable prediction service
  - Decoupled agronomic advisory rules
         │
         ▼
PHASE 3 (Future Field Hardening):
  - External zero-shot benchmark against PlantDoc
  - Mixed fine-tuning (PlantVillage 80% + Field 20%)
  - Training lightweight binary leaf detector (YOLOv8-Nano)
```
