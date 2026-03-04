"""
Gunicorn 多 worker 配置：根据 GPU 数量启动多个服务端，每个 worker 绑定一张 GPU。
使用方式: gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py launch_OralAgent:OralAgent
"""
import os
from pathlib import Path
from datetime import datetime
from gpu_utils import get_gpu_count, get_recommended_workers

# 请求日志根目录，与 launch_OralAgent 中 DEFAULT_REQUEST_LOG_DIR 一致
_REQUEST_LOG_DIR = os.getenv("ORALAGENT_REQUEST_LOG_DIR", "logs/requests")


def on_starting(server):
    """Master 进程启动时设置统一的请求日志 session 目录，并预先创建目录，保证多 worker 共用一个文件夹。"""
    session_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["ORALAGENT_REQUEST_LOG_SESSION_DIR"] = session_dir
    if _REQUEST_LOG_DIR:
        log_path = Path(_REQUEST_LOG_DIR) / session_dir
        log_path.mkdir(parents=True, exist_ok=True)
        if os.environ.get("ORAL_AGENT_DEBUG"):
            print(f"[OralAgent] request log session dir: {log_path}")

# 可选：每个 agent 实例预估显存（GB），用于按显存推算 worker 数
_VRAM_GB = os.environ.get("ORAL_AGENT_VRAM_PER_AGENT_GB")
VRAM_PER_AGENT_GB = float(_VRAM_GB) if _VRAM_GB else None

# 单卡允许的 worker 数，默认 1；设为 >1 即单卡多 worker（注意显存，避免 OOM）
# 由环境变量 ORAL_AGENT_MAX_WORKERS_PER_GPU 控制，get_recommended_workers 内部会读
num_gpus = get_gpu_count()
workers = get_recommended_workers(
    num_gpus=num_gpus,
    vram_per_agent_gb=VRAM_PER_AGENT_GB,
)

def post_fork(server, worker):
    """每个 worker 进程 fork 后只看到一张 GPU，避免多进程抢同一张卡。用 PID 取模分配。"""
    gpu_id = (worker.pid % num_gpus) if num_gpus else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if os.environ.get("ORAL_AGENT_DEBUG"):
        print(f"[OralAgent] worker pid={worker.pid} -> GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")


bind = "0.0.0.0:8124"
worker_class = "uvicorn.workers.UvicornWorker"
workers = workers
threads = 1
timeout = 300
loglevel = os.environ.get("ORAL_AGENT_LOG_LEVEL", "info")
