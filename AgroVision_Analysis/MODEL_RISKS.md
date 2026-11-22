# Technical & Agricultural Risk Assessment

**Author:** Senior Machine Learning Engineer, AI Researcher & Technical Architect  
**Project:** AgroVision — AI-Powered Crop Disease Screening & Advisory System  
**Date:** November 2026  
**Status:** Risk & Safety Analysis  

---

## 1. Executive Summary

Deploying computer vision models in agriculture carries distinct real-world risks. Unlike consumer image classification (e.g., classifying dog breeds), an erroneous crop disease diagnosis can lead to:
- Inappropriate application of chemical pesticides or fungicides (economic loss, soil toxicity, chemical resistance).
- Failure to arrest highly contagious pathogens (e.g., *Phytophthora infestans* / Potato Late Blight), resulting in total crop destruction.
- Farmer distrust of AI-assisted agronomic tools.

This document identifies the technical failure modes of the underlying models and formulates actionable architectural mitigations.

---

## 2. Technical Limitations & Failure Modes

### 2.1 The PlantVillage Domain-Shift Gap (Laboratory vs. Field Conditions)
Both models were trained on the **PlantVillage** dataset. In academic literature (e.g., Mohanty et al., 2016; Hughes & Salathé, 2015; Ferentinos, 2018), PlantVillage models routinely achieve $>99\%$ accuracy in-domain, but experience severe performance degradation (often dropping to $30\% - 50\%$ accuracy) when evaluated on in-situ agricultural field imagery (e.g., PlantDoc dataset).

```
PlantVillage Domain (Training):                Real Agricultural Field (Deployment):
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  [Uniform Gray/Black Paper]  │              │  [Overlapping Weeds, Soil]   │
│         ╭─────────╮          │              │    \     ╭─────────╮   /      │
│         │ Detached│          │              │     \    │ Connected│  /       │
│         │  Leaf   │          │  ───────►    │      \───│  Foliage │ /        │
│         ╰─────────╯          │  Domain Gap  │          ╰─────────╯           │
│   [Diffused Studio Light]    │              │  [Harsh Glare, Rain, Shadows]│
└──────────────────────────────┘              └──────────────────────────────┘
```

#### Specific Field Confounders:
1. **Background Clutter:** In the field, leaves appear against soil, mulching plastic, weeds, dry debris, or neighboring plants. Standard CNNs often learn spurious background correlations present in laboratory benches rather than pure lesion morphology.
2. **Dynamic Natural Lighting:** Direct midday sun causes specular reflections (glare) that bleach lesion colors. Dappled canopy shadows create false necrotic patterns.
3. **Wetness & Droplets:** Morning dew or irrigation droplets act as miniature lenses, distorting leaf texture and creating false reflections that can be misclassified as fungal pustules or powdery mildew.
4. **Camera Hardware Heterogeneity:** Smallholder farmers use low-cost mobile devices with varying chromatic aberration, lens distortion, sensor noise, and autofocus failures.

---

### 2.2 The Closed-World Softmax Fallacy & Out-of-Distribution (OOD) Inputs
The trained models operate under a **closed-world assumption**: given any 3-channel $224 \times 224$ input tensor, the final softmax layer forces the output probabilities to sum to 1.0 across the 38 pre-defined classes.

#### Failure Modes:
1. **Non-Leaf Image Uploads:** If a farmer inadvertently uploads a photo of an insect pest, a tractor, their hand, or a dog, the model will output a high-confidence prediction for one of the 38 crop disease classes.
2. **Unsupported Crops:** If a farmer photographs a diseased Rice, Wheat, Cassava, or Cotton leaf (none of which are among the 14 supported crops), the system will force the prediction into an unsupported class (e.g., classifying Rice Blast as Corn Northern Leaf Blight).
3. **Unseen Pathologies:** If a supported crop (e.g., Tomato) contracts an unsupported disease (e.g., Tomato Pith Necrosis or Verticillium Wilt), the model cannot report "Unknown Disease" unless an explicit out-of-distribution detector or abstention threshold is active.

---

### 2.3 Single-Label Multi-Class Assumption (Co-Infections)
In nature, crops frequently suffer from:
- Simultaneous fungal infection and insect damage (e.g., Tomato Early Blight accompanied by Spider Mite feeding punctures).
- Secondary opportunistic infections invading necrotic tissue caused by physical hail or wind damage.
- Abiotic stress (e.g., potassium deficiency causing leaf-edge scorch) mimicking fungal leaf scorch.

Because the models employ single-label softmax cross-entropy, they can predict only the single dominant class and are architecturally blind to co-infections.

---

### 2.4 The Contextual Agricultural Data Fallacy (Architectural Warning)

> [!CAUTION]
> **Architectural Mandate:** External environmental parameters (temperature, humidity, rainfall, dew point, soil pH) must **NEVER** be naively appended to the intermediate CNN feature vectors (e.g., concatenating weather scalars into the MobileNetV2 1280-d pooling vector) during this phase.

#### Why Naive Multimodal Concat Fails Here:
1. **Dataset Unalignment:** The training dataset (PlantVillage) contains **zero** paired environmental telemetry. If a developer creates a 1284-dimensional dense layer ($1280 \text{ visual} + 4 \text{ weather}$) and feeds synthetic or zeroed weather data, the network will learn corrupted representations.
2. **Spurious Invariant Correlations:** Pathogens require specific epidemiological conditions (e.g., *Phytophthora infestans* requires $>90\%$ humidity and temperatures between $15^\circ\text{C} - 22^\circ\text{C}$). If weather data is directly backpropagated through image weights without rigorous multimodal training on hundreds of thousands of aligned field samples, the model will hallucinate disease predictions based solely on hot or rainy weather, even when viewing a pristine healthy leaf.
3. **Correct Architectural Pattern:** Environmental data must remain **strictly contextual** in the initial AgroVision architecture. It must reside in a downstream **Agronomic Decision Engine (Rule-Based or Bayesian Expert System)** that validates, contextualizes, and refines the recommendation *after* the vision model outputs its calibrated visual prediction.

```
CORRECT ARCHITECTURE: Decoupled Visual & Agronomic Pipelines

   [Uploaded Leaf Image]                       [User Context / Weather Telemetry]
            │                                             │
            ▼                                             │
   ┌───────────────────┐                                  │
   │ Computer Vision   │                                  │
   │ Model (ml/)       │                                  │
   └────────┬──────────┘                                  │
            │ Calibrated Visual Prediction                │ Location, Temp, Humidity, Rainfall
            ▼                                             ▼
   ┌─────────────────────────────────────────────────────────────┐
   │         AgroVision Agronomic Context & Advisory Engine      │
   │  - Cross-references visual prediction with weather risk    │
   │  - Validates if conditions support pathogen development     │
   │  - Enriches diagnosis with customized treatment actions    │
   └──────────────────────────────┬──────────────────────────────┘
                                  ▼
                     [Enriched Farmer Diagnosis]
```

---

## 3. Risk Mitigation & System Improvement Roadmap

To ensure reliable operation in actual agricultural environments, the AgroVision platform must implement the following safeguards:

### 3.1 Tier 1: Two-Stage Input Quality Gate (Pre-Inference)
Before an uploaded image reaches the MobileNetV2 disease classifier:
1. **Image Quality Filter:** Calculate Laplacian variance to detect severe motion blur ($\text{Var}(\Delta I) < \tau_{\text{blur}}$) and reject unreadable photos.
2. **Leaf Presence Verification (Greenness / Plant Masking):** Compute color space distributions (e.g., Excess Green Index $2G - R - B$ or HSV green hue clustering) or deploy a tiny binary MobileNetV3 / YOLOv8 leaf-detector to verify the presence of agricultural foliage before disease inference.

### 3.2 Tier 2: Calibrated Selective Prediction (Inference Gating)
Enforce the temperature-calibrated abstention threshold from `ml/`:
- If $\max_i \hat{p}_i < 0.30$: The API suppresses the diagnosis, marks status as `ABSTAIN`, and returns:
  *"The leaf pattern is ambiguous or unsupported. Please capture a closer, sharper photograph under diffused natural light, or consult a local agricultural extension officer."*

### 3.3 Tier 3: Agronomic Context & Plausibility Validation (Post-Inference)
Deploy an expert validation matrix cross-referencing visual diagnoses with contextual parameters:
- *Example:* If the visual model predicts **Late Blight** on Potato with 65% confidence, but the telemetry reports **Ambient Temperature = $38^\circ\text{C}$** and **Relative Humidity = $22\%$**, Late Blight pathogen development is biologically implausible (sporulation ceases above $27^\circ\text{C}$). The system flags this discrepancy:
  *"Visual symptoms resemble Late Blight, but current high temperatures make active blight unlikely. Check for heat stress or drought scorch."*

### 3.4 Tier 4: Future Multimodal & Field Retraining
1. **Field Dataset Integration:** Leverage the existing `--external_dir` code in `ml/train_agrovision.py` to fine-tune the model on the PlantDoc dataset (containing complex, in-situ farm photos).
2. **True Multimodal Cross-Attention:** When a properly curated, geographically aligned dataset with paired sensor telemetry is collected, train a unified vision-environmental cross-attention transformer.
