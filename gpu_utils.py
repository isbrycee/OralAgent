"""
GPU 检测与多实例数量计算，用于多 worker 部署。
不在主进程 import torch，避免 fork 后 CUDA 上下文问题。
"""
import os
import subprocess


def get_gpu_count() -> int:
    """通过 nvidia-smi 获取 GPU 数量，不在本模块内 import torch。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return len([x for x in r.stdout.strip().split("\n") if x.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def get_gpu_memory_gb() -> list[float]:
    """获取每张 GPU 的显存大小（GB）。返回列表，长度为 GPU 数量。"""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return [float(x.strip()) / 1024.0 for x in r.stdout.strip().split("\n")]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


def get_recommended_workers(
    *,
    num_gpus: int | None = None,
    vram_per_agent_gb: float | None = None,
    max_workers_per_gpu: int | None = None,
) -> int:
    """
    根据显存或 GPU 数量计算推荐的 worker 数量。

    - 若设置了 ORAL_AGENT_WORKERS：直接使用该值。
    - 否则若设置了 ORAL_AGENT_VRAM_PER_AGENT_GB：按总显存/单实例预估显存计算，
      且不超过 num_gpus * max_workers_per_gpu。
    - 否则：worker 数 = GPU 数 * max_workers_per_gpu（支持单卡多 worker）。

    max_workers_per_gpu 可通过环境变量 ORAL_AGENT_MAX_WORKERS_PER_GPU 覆盖（默认 1）。
    """
    if num_gpus is None:
        num_gpus = get_gpu_count()
    if num_gpus <= 0:
        return 1

    if max_workers_per_gpu is None:
        try:
            max_workers_per_gpu = max(1, int(os.environ.get("ORAL_AGENT_MAX_WORKERS_PER_GPU", "1")))
        except (ValueError, TypeError):
            max_workers_per_gpu = 1

    env_workers = os.environ.get("ORAL_AGENT_WORKERS")
    if env_workers is not None:
        try:
            return max(1, int(env_workers))
        except ValueError:
            pass

    if vram_per_agent_gb is not None and vram_per_agent_gb > 0:
        mems = get_gpu_memory_gb()
        if mems:
            total_gb = sum(mems)
            by_vram = max(1, int(total_gb / vram_per_agent_gb))
            return min(by_vram, num_gpus * max_workers_per_gpu)

    return num_gpus * max_workers_per_gpu
