"""
CPU Performance Benchmark for AgroVision ML Model.
Measures model load latency, preprocessing latency, forward-pass latency,
total pipeline latency, and process memory consumption.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gc
import json
import os
import platform
import time
import numpy as np
from PIL import Image
import psutil
import torch

from backend.ml.predictor import CropDiseasePredictor
from backend.ml.model_loader import AgroVisionModelLoader
from backend.ml.preprocessing import ImagePreprocessor


def run_benchmark(num_warmup: int = 10, num_runs: int = 100):
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Measure Model Load Time
    t0 = time.perf_counter()
    loader = AgroVisionModelLoader()
    model, metadata = loader.load_model(force_reload=True)
    t_load = (time.perf_counter() - t0) * 1000.0  # ms

    mem_after_load_mb = process.memory_info().rss / (1024 * 1024)
    model_mem_delta_mb = mem_after_load_mb - mem_before_mb

    preprocessor = ImagePreprocessor(img_size=metadata.input_size)
    predictor = CropDiseasePredictor()

    # Synthetic realistic image
    arr = np.zeros((400, 400, 3), dtype=np.uint8)
    arr[:, :, 1] = 160
    arr[::4, :] = [80, 180, 40]
    arr[:, ::4] = [50, 130, 20]
    test_img = Image.fromarray(arr)

    # Pre-warmup
    for _ in range(num_warmup):
        _ = predictor.predict(test_img)

    prep_times = []
    infer_times = []
    total_times = []

    # Repeated timed runs
    for _ in range(num_runs):
        t_start = time.perf_counter()

        # Preprocessing alone
        t_p0 = time.perf_counter()
        tensor = preprocessor.preprocess(test_img)
        t_prep = (time.perf_counter() - t_p0) * 1000.0
        prep_times.append(t_prep)

        # Inference alone (model forward pass + temperature calibration + softmax)
        t_i0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            scaled = logits / metadata.temperature
            probs = torch.softmax(scaled, dim=1)
        t_infer = (time.perf_counter() - t_i0) * 1000.0
        infer_times.append(t_infer)

        t_total = (time.perf_counter() - t_start) * 1000.0
        total_times.append(t_total)

    mem_peak_mb = process.memory_info().rss / (1024 * 1024)

    results = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cpu_only": not torch.cuda.is_available(),
        },
        "model_loading": {
            "load_time_ms": round(t_load, 2),
            "baseline_process_mem_mb": round(mem_before_mb, 2),
            "post_load_process_mem_mb": round(mem_after_load_mb, 2),
            "model_memory_delta_mb": round(model_mem_delta_mb, 2),
            "peak_process_mem_mb": round(mem_peak_mb, 2),
        },
        "latencies_ms": {
            "runs": num_runs,
            "warmup_runs": num_warmup,
            "preprocessing": {
                "mean": round(float(np.mean(prep_times)), 2),
                "median": round(float(np.median(prep_times)), 2),
                "p90": round(float(np.percentile(prep_times, 90)), 2),
                "p95": round(float(np.percentile(prep_times, 95)), 2),
                "min": round(float(np.min(prep_times)), 2),
                "max": round(float(np.max(prep_times)), 2),
            },
            "inference": {
                "mean": round(float(np.mean(infer_times)), 2),
                "median": round(float(np.median(infer_times)), 2),
                "p90": round(float(np.percentile(infer_times, 90)), 2),
                "p95": round(float(np.percentile(infer_times, 95)), 2),
                "min": round(float(np.min(infer_times)), 2),
                "max": round(float(np.max(infer_times)), 2),
            },
            "total_pipeline": {
                "mean": round(float(np.mean(total_times)), 2),
                "median": round(float(np.median(total_times)), 2),
                "p90": round(float(np.percentile(total_times, 90)), 2),
                "p95": round(float(np.percentile(total_times, 95)), 2),
                "min": round(float(np.min(total_times)), 2),
                "max": round(float(np.max(total_times)), 2),
            },
        },
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_benchmark(num_warmup=10, num_runs=100)
