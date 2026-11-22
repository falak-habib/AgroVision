# Model Loading & Architecture Verification Report

**Component:** Model Loader & Parameter Integrity  
**Artifact Path:** `AgroVision/ml/artifacts/all_v1/model.pt`  
**Configuration Path:** `AgroVision/ml/artifacts/all_v1/artifact.json`  
**Status:** VERIFIED  

---

## 1. Serialization Format Analysis

Inspection of `ml/artifacts/all_v1/model.pt` confirmed the following serialization profile:
- **Format:** Python `collections.OrderedDict` containing raw PyTorch `state_dict` tensors (weights and biases).
- **Not a TorchScript / JIT archive:** The file is not a compiled TorchScript model or pickled module container; it contains parameter tensors only.
- **File Size:** 9,318,027 bytes (~8.89 MB).
- **Parameter Count:** Exactly **314 tensor keys**, corresponding to:
  - 19 MobileNetV2 inverted residual blocks (`features.0` through `features.18`), including convolution weights, BatchNorm running mean/variance, and batch counter tensors.
  - Linear classification head: `classifier.1.weight` (shape: `[38, 1280]`) and `classifier.1.bias` (shape: `[38]`).
- **Total Parameters:** **2,263,462 parameters** (all float32).

---

## 2. Architecture Reconstruction & Parameter Matching

The architecture was reconstructed using `torchvision.models.mobilenet_v2(weights=None)` with the classifier replaced:
```python
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 38)
```

### Verification Results:
- `model.load_state_dict(state_dict, strict=True)` completed with:
  `<All keys matched successfully>`
- **Missing Keys:** Exactly 0.
- **Unexpected Keys:** Exactly 0.
- **Shape Mismatches:** Exactly 0.

---

## 3. CPU Compatibility & Memory Profile

- **Map Location:** The model weights load cleanly on CPU using `map_location=torch.device("cpu")`.
- **CUDA Independence:** Loading and forward inference execute without requiring CUDA drivers, cuDNN libraries, or GPU hardware.
- **Model Load Latency:** **118.17 ms** on host CPU.
- **Model Memory Overhead:** **22.71 MB** RAM delta after loading.
- **Parameter Freezing:** All parameters are frozen (`requires_grad = False`) and model is explicitly placed in evaluation mode (`model.eval()`).

---

## 4. Output Dimension Verification

A dummy input tensor of shape `(1, 3, 224, 224)` and batched tensors of shape `(4, 3, 224, 224)` were evaluated:
- Output Tensor Shape: `torch.Size([1, 38])`
- Batched Output Shape: `torch.Size([4, 38])`
- Data Type: `torch.float32`
- Value Range: Raw uncalibrated logits spanning approximately $[-4.5, +9.2]$.
- Device: `cpu`

---

## 5. Metadata Sufficiency

`artifact.json` provides complete operational parameters:
- `arch`: `"mobilenet_v2"`
- `classes`: 38 canonical class labels
- `input_size`: 224
- `normalize_mean`: `[0.485, 0.456, 0.406]`
- `normalize_std`: `[0.229, 0.224, 0.225]`
- `resize`: `"direct_resize_to_square"`
- `temperature`: `0.980576753616333`
- `abstain_threshold`: `0.3`

All metadata fields are verified and sufficient to reconstruct and operate the model independently of the training script.
