#!/usr/bin/env bash
# 根据本机 GPU 数量启动多个 OralAgent worker，每个 worker 绑定一张 GPU。
# 需要先安装: pip install gunicorn "uvicorn[standard]"
#
# 可选环境变量:
#   ORAL_AGENT_WORKERS=4                    强制 worker 数量
#   ORAL_AGENT_MAX_WORKERS_PER_GPU=2        单卡 worker 数（默认 1），单卡多 worker 时设大
#   ORAL_AGENT_VRAM_PER_AGENT_GB=12         按显存估算 worker 数（总显存/该值）
#   ORAL_AGENT_DEBUG=1                     打印每个 worker 绑定的 GPU
#   ORAL_AGENT_LOG_LEVEL=info               gunicorn 日志级别

set -e
cd "$(dirname "$0")"

if ! command -v gunicorn &>/dev/null; then
  echo "请先安装 gunicorn: pip install gunicorn \"uvicorn[standard]\""
  exit 1
fi

exec gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py launch_OralAgent:OralAgent
