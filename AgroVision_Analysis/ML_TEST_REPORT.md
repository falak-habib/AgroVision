# Automated ML Integration Test Report

**Execution Date:** November 22, 2026  
**Test Suite:** `backend/ml/tests/run_all_tests.py`  
**Test Framework:** Python `unittest`  
**Overall Result:** 31 PASSED, 0 FAILED, 0 ERRORS (100% Pass Rate)  
**Execution Duration:** 1.340 seconds  

---

## 1. Test Execution Summary

```
============================================================
AGROVISION ML TEST SUMMARY
Ran:      31 tests
Passed:   31
Failures: 0
Errors:   0
Status:   ALL TESTS PASSING
============================================================
```

---

## 2. Granular Test Case Breakdown

### Module 1: Model Loader Tests (`test_model_loader.py`)
| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_load_metadata` | Verifies all required fields in `artifact.json` (arch, classes, input_size, mean, std, T, threshold). | **PASS** |
| `test_load_model_on_cpu` | Verifies model loads cleanly on CPU, weights are frozen, and no CUDA calls occur. | **PASS** |
| `test_output_dimensions` | Verifies single dummy input produces tensor of exact shape `(1, 38)`. | **PASS** |
| `test_batch_output_dimensions` | Verifies batched input produces tensor of shape `(4, 38)`. | **PASS** |
| `test_missing_artifact_handling`| Verifies `ModelArtifactError` is raised when artifact path is invalid. | **PASS** |

### Module 2: Preprocessing Tests (`test_preprocessing.py`)
| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_pil_image_preprocessing` | Verifies PIL Image transforms to normalized float32 tensor of shape `(1, 3, 224, 224)`. | **PASS** |
| `test_rgba_image_converted_to_rgb`| Verifies 4-channel PNG RGBA is converted to 3-channel RGB. | **PASS** |
| `test_grayscale_image_converted_to_rgb`| Verifies 1-channel grayscale is converted to 3-channel RGB. | **PASS** |
| `test_normalization_values` | Verifies mathematical exactness of ImageNet mean and std subtraction. | **PASS** |
| `test_invalid_input_type` | Verifies `ImageValidationError` is raised when input is not an image or bytes. | **PASS** |

### Module 3: Class Mapping Tests (`test_class_mapping.py`)
| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_canonical_classes_count` | Verifies exactly 38 classes in canonical list. | **PASS** |
| `test_cross_check_with_artifact_json`| Verifies exact 1:1 match in naming and ordering against `artifact.json`. | **PASS** |
| `test_every_class_index_0_to_37` | Tests retrieving all 38 indices (0 to 37), ensuring crop, disease, and health status parse cleanly. | **PASS** |
| `test_healthy_classes_count` | Verifies exactly 12 healthy classes are identified across supported crops. | **PASS** |
| `test_supported_crops_count` | Verifies exactly 14 unique botanical crops are supported. | **PASS** |
| `test_out_of_bounds_index` | Verifies `IndexError` on index $< 0$ or $> 37$. | **PASS** |
| `test_label_lookup` | Verifies reverse lookup from canonical string label to ClassInfo. | **PASS** |

### Module 4: Calibration & Abstention Tests (`test_calibration_abstention.py`)
| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_temperature_scaling_mathematics`| Verifies $p = \text{softmax}(z / T)$ sharpens distribution when $T < 1.0$ and preserves argmax. | **PASS** |
| `test_abstention_above_threshold` | Verifies confidence $\ge 0.30$ returns `"PREDICTION_AVAILABLE"`. | **PASS** |
| `test_abstention_below_threshold` | Verifies uniform/low confidence $< 0.30$ returns `"ABSTAIN"`. | **PASS** |
| `test_abstention_boundary_conditions`| Tests boundary conditions: $0.3000$ (accepted), $0.3001$ (accepted), $0.2999$ (abstained). | **PASS** |

### Module 5: Image Validator Tests (`test_image_validator.py`)
| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_valid_sharp_image` | Sharp, well-lit image passes validation cleanly. | **PASS** |
| `test_corrupted_bytes` | Random non-image bytes intercepted with `CORRUPT_OR_UNREADABLE_IMAGE`. | **PASS** |
| `test_low_resolution_image` | Micro-thumbnail ($32 \times 32$) intercepted with `IMAGE_RESOLUTION_TOO_LOW`. | **PASS** |
| `test_severely_blurry_image` | Gaussian blurred image (low Laplacian variance) intercepted with `IMAGE_TOO_BLURRY`. | **PASS** |
| `test_extremely_dark_image` | Near-black image (luminance $< 25$) intercepted with `IMAGE_TOO_DARK`. | **PASS** |
| `test_extremely_bright_image` | Washed-out/overexposed image intercepted with `IMAGE_TOO_BRIGHT`. | **PASS** |
| `test_leaf_presence_heuristic_does_not_reject_brown_leaf` | Verifies necrotic brown leaf (low green) is NOT rejected. | **PASS** |

### Module 6: Predictor Integration Tests (`test_predictor_integration.py`)
| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_predict_sharp_synthetic_image` | End-to-end forward pass returning structured prediction with top-3 candidates. | **PASS** |
| `test_predict_blurry_image_returns_invalid_image` | End-to-end pipeline intercepts blurry image before forward pass. | **PASS** |
| `test_predict_abstention_enforcement` | High threshold ($0.999$) triggers `"status": "ABSTAIN"` with referral message. | **PASS** |

---

## 3. Heuristic vs. Rigorous Test Classification

| Test Scope | Verification Methodology | Nature of Check |
| :--- | :--- | :--- |
| **Model Weights & CPU Inference** | PyTorch C++ linear algebra execution | **Mathematically Rigorous** |
| **Class Index Mapping (0–37)** | Exact string parity with `artifact.json` | **Exact Deterministic Verification** |
| **Preprocessing Normalization** | Floating-point arithmetic matching | **Exact Deterministic Verification** |
| **Temperature Calibration** | Softmax logit division verification | **Mathematically Rigorous** |
| **Abstention Boundary** | Strict float comparison ($\text{conf} < \theta$) | **Exact Deterministic Verification** |
| **Blur & Darkness Detection** | Laplacian convolution & luminance mean | **Empirical Heuristic Screening** |
| **Foliage Presence** | Excess Green Index ($2G - R - B$) | **Diagnostic Heuristic Only (Does Not Reject)** |

---

## 4. Unresolved Issues

None. All 31 tests are passing with zero failures and zero errors.
