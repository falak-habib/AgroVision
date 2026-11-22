# CPU Performance Benchmark Report

**Component:** AgroVision ML Predictor CPU Serving  
**Date:** November 2026  
**Status:** EMPIRICALLY BENCHMARKED (100 Repetitions)  

---

## 1. Benchmark Execution Environment & Hardware Specifications

All benchmark measurements were captured empirically using `backend/ml/tests/benchmark_cpu.py` on the local host machine under real operating conditions:

| Parameter | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Pro (`Windows-11-10.0.26300-SP0`) |
| **Processor (CPU)** | AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD |
| **CPU Topology** | 8 Physical Cores, 16 Logical Threads |
| **Total Host RAM** | 31.28 GB |
| **Python Version** | 3.13.5 (64-bit) |
| **PyTorch Runtime** | PyTorch 2.15.0+cpu (`torchvision` 0.29.2+cpu) |
| **Execution Mode** | **Strict CPU Only** (`torch.cuda.is_available() == False`) |

---

## 2. Latency Metrics Across 100 Repetitions

Following 10 warm-up iterations, 100 consecutive prediction cycles were executed on a standardized input image:

| Stage | Mean Latency | Median Latency | Min Latency | P90 Latency | P95 Latency | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image Preprocessing** | 2.48 ms | **2.39 ms** | 2.00 ms | 2.76 ms | 3.02 ms | 4.06 ms |
| **Model Inference (Forward + Softmax + Temp)** | 20.37 ms | **19.65 ms** | 15.93 ms | 23.24 ms | 24.46 ms | 33.28 ms |
| **Total Pipeline (End-to-End)** | 22.86 ms | **22.03 ms** | 18.08 ms | 25.96 ms | **27.14 ms** | 35.69 ms |

```
Latency Distribution (ms):
[Min: 18.08] ─── [P50/Median: 22.03] ─── [Mean: 22.86] ─── [P95: 27.14] ─── [Max: 35.69]
```

---

## 3. Memory & Resource Consumption Profile

Measurements were collected via `psutil` tracking resident set size (RSS):

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Process Baseline RAM** | 285.56 MB | Python runtime + loaded PyTorch C++ libraries |
| **Post-Model-Load RAM** | 308.27 MB | Full MobileNetV2 weights loaded in memory |
| **Net Model Memory Delta** | **22.71 MB** | Ultra-compact memory footprint |
| **Peak RAM During 100 Inferences** | **345.94 MB** | Peak memory including intermediate activation tensors |
| **Model Loading Duration** | **118.17 ms** | Cold-start weight initialization time |

---

## 4. Throughput & Production Capacity Analysis

- **Single-Core Theoretical Throughput:**
  $$\text{Throughput} = \frac{1000\text{ ms}}{22.03\text{ ms/request}} \approx \mathbf{45.4\text{ requests/second}}$$
- **Uvicorn / FastAPI Concurrency Assessment:**
  On a 2-vCPU or 4-vCPU container instance, running 2–4 Uvicorn worker processes will easily deliver **80–150 requests/second** sustained throughput with sub-50ms response times.
- **Premature Optimization Notice:**
  Given the median latency of 22 ms, converting to ONNX Runtime or TorchScript is unnecessary for Phase 2, but remains an available optimization for future edge deployment.
