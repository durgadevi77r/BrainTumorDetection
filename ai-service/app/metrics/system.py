"""
app/metrics/system.py — System resource metrics collection.

Collects CPU, RAM, disk, GPU (if available), uptime, and API latency
statistics using psutil.  All collectors are non-fatal: if a metric
cannot be read the field is set to None or a safe default.

Usage
-----
    from app.metrics.system import get_system_metrics

    data = get_system_metrics()
    print(data["cpu_percent"], data["ram_used_mb"])
"""

from __future__ import annotations

import os
import platform
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil

# ── Module-level startup timestamp ───────────────────────────────────────────
_SERVICE_START_TIME: float = time.time()


def _uptime_seconds() -> float:
    """Seconds since the ai-service process started."""
    return round(time.time() - _SERVICE_START_TIME, 1)


def _cpu_metrics() -> Dict[str, Any]:
    """Return per-core and overall CPU utilisation."""
    try:
        overall = psutil.cpu_percent(interval=0.1)
        per_core: List[float] = psutil.cpu_percent(interval=0.1, percpu=True)  # type: ignore[assignment]
        freq = psutil.cpu_freq()
        return {
            "cpu_percent": round(overall, 1),
            "cpu_per_core": [round(c, 1) for c in per_core],
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq_mhz": round(freq.current, 0) if freq else None,
        }
    except Exception:
        return {
            "cpu_percent": None,
            "cpu_per_core": [],
            "cpu_count_logical": None,
            "cpu_count_physical": None,
            "cpu_freq_mhz": None,
        }


def _ram_metrics() -> Dict[str, Any]:
    """Return virtual memory statistics in MB."""
    try:
        vm = psutil.virtual_memory()
        return {
            "ram_total_mb": round(vm.total / 1024 / 1024, 1),
            "ram_used_mb": round(vm.used / 1024 / 1024, 1),
            "ram_available_mb": round(vm.available / 1024 / 1024, 1),
            "ram_percent": round(vm.percent, 1),
        }
    except Exception:
        return {
            "ram_total_mb": None,
            "ram_used_mb": None,
            "ram_available_mb": None,
            "ram_percent": None,
        }


def _disk_metrics() -> Dict[str, Any]:
    """Return disk usage for the filesystem containing the project."""
    try:
        # Use the current working directory as the mount point
        path = os.path.abspath(".")
        du = psutil.disk_usage(path)
        return {
            "disk_total_gb": round(du.total / 1024 / 1024 / 1024, 2),
            "disk_used_gb": round(du.used / 1024 / 1024 / 1024, 2),
            "disk_free_gb": round(du.free / 1024 / 1024 / 1024, 2),
            "disk_percent": round(du.percent, 1),
        }
    except Exception:
        return {
            "disk_total_gb": None,
            "disk_used_gb": None,
            "disk_free_gb": None,
            "disk_percent": None,
        }


def _gpu_metrics() -> Dict[str, Any]:
    """
    Return GPU metrics if TensorFlow can see a GPU.

    Falls back gracefully when no GPU is available or when
    nvidia-ml-py / tf GPU info is unavailable.
    """
    gpu_info: List[Dict[str, Any]] = []

    # Try PyTorch CUDA device list (zero-cost if torch is already imported)
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "memory_used_mb": None,
                "memory_total_mb": round(props.total_memory / (1024 ** 2)),
                "utilization_percent": None,
            })
    except Exception:
        pass

    # Enhance with nvidia-smi data if pynvml is available
    try:
        import pynvml  # type: ignore[import]
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            entry = {
                "index": i,
                "name": pynvml.nvmlDeviceGetName(handle),
                "memory_used_mb": round(mem.used / 1024 / 1024, 1),
                "memory_total_mb": round(mem.total / 1024 / 1024, 1),
                "utilization_percent": util.gpu,
            }
            # Replace the TF entry if present
            if i < len(gpu_info):
                gpu_info[i] = entry
            else:
                gpu_info.append(entry)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return {
        "gpu_available": len(gpu_info) > 0,
        "gpu_count": len(gpu_info),
        "gpus": gpu_info,
    }


def _process_metrics() -> Dict[str, Any]:
    """Return metrics for the current Python process."""
    try:
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        return {
            "process_pid": proc.pid,
            "process_cpu_percent": round(proc.cpu_percent(interval=0.05), 1),
            "process_ram_mb": round(mem_info.rss / 1024 / 1024, 1),
            "process_threads": proc.num_threads(),
        }
    except Exception:
        return {
            "process_pid": None,
            "process_cpu_percent": None,
            "process_ram_mb": None,
            "process_threads": None,
        }


def get_system_metrics() -> Dict[str, Any]:
    """
    Collect and return a snapshot of system resource metrics.

    Returns
    -------
    dict with keys:
        timestamp, uptime_seconds,
        cpu_percent, cpu_per_core, cpu_count_logical, cpu_count_physical, cpu_freq_mhz,
        ram_total_mb, ram_used_mb, ram_available_mb, ram_percent,
        disk_total_gb, disk_used_gb, disk_free_gb, disk_percent,
        gpu_available, gpu_count, gpus,
        process_pid, process_cpu_percent, process_ram_mb, process_threads,
        platform, python_version
    """
    metrics: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": _uptime_seconds(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
    }
    metrics.update(_cpu_metrics())
    metrics.update(_ram_metrics())
    metrics.update(_disk_metrics())
    metrics.update(_gpu_metrics())
    metrics.update(_process_metrics())
    return metrics
