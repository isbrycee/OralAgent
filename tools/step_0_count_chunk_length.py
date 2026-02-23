import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

HYBRID_AUTO_DIR = "hybrid_auto"
STEP3_LLM_CLEANED_SUFFIX = "_step3_llm_cleaned.json"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_token_counter() -> Optional[Callable[[str], int]]:
    """返回用于计算 token 数量的函数，若 tiktoken 未安装则返回 None。"""
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        enc = tiktoken.get_encoding("cl100k_base")  # GPT-4/3.5 使用的编码
        return lambda s: len(enc.encode(s))
    except Exception:
        return None


def discover_step3_llm_cleaned_files(root_dir: Path) -> List[Path]:
    """
    在根目录下查找所有 hybrid_auto/*_step3_llm_cleaned.json 文件。
    目录结构：root_dir / 类别 / 语言目录(如1_CH、1_EN) / 书名 / hybrid_auto / *_step3_llm_cleaned.json
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        return []
    found: List[Path] = []
    for category_dir in root.iterdir():
        if not category_dir.is_dir():
            continue
        for lang_dir in category_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            for book_dir in lang_dir.iterdir():
                if not book_dir.is_dir():
                    continue
                hybrid_dir = book_dir / HYBRID_AUTO_DIR
                if not hybrid_dir.is_dir():
                    continue
                for f in hybrid_dir.iterdir():
                    if f.is_file() and f.name.endswith(STEP3_LLM_CLEANED_SUFFIX):
                        found.append(f)
    return sorted(found)


def get_length_bucket(length: int, bucket_edges: List[int]) -> str:
    """将长度映射到区间标签，如 '0-50', '51-100', '101-200' 等。"""
    for i, edge in enumerate(bucket_edges):
        if length < edge:
            prev = bucket_edges[i - 1] if i > 0 else 0
            return f"{prev}-{edge - 1}"
    return f"{bucket_edges[-1]}+"


def compute_length_distribution(lengths: List[int], bucket_edges: List[int]) -> Dict[str, Any]:
    """计算长度列表的分布统计和分桶直方图。"""
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "histogram": {}}
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    buckets: Dict[str, int] = defaultdict(int)
    for L in lengths:
        label = get_length_bucket(L, bucket_edges)
        buckets[label] += 1
    hist_items = []
    prev = 0
    for edge in bucket_edges:
        label = f"{prev}-{edge - 1}"
        hist_items.append((label, buckets[label]))
        prev = edge
    hist_items.append((f"{bucket_edges[-1]}+", buckets[f"{bucket_edges[-1]}+"]))
    return {
        "count": n,
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / n,
        "median": sorted_lens[n // 2] if n % 2 == 1 else (sorted_lens[n // 2 - 1] + sorted_lens[n // 2]) / 2,
        "histogram": dict(hist_items),
    }


def _print_distribution(title: str, dist: Dict[str, Any]) -> None:
    """打印单种分布的统计与直方图。"""
    print("=" * 50)
    print(title)
    print("=" * 50)
    print(f"  条目数: {dist['count']}")
    if dist["count"] > 0:
        print(f"  最小值: {dist['min']}")
        print(f"  最大值: {dist['max']}")
        print(f"  平均值: {dist['mean']:.2f}")
        print(f"  中位数: {dist['median']}")
        print("  分桶直方图:")
        print(f"    {'区间':>14}  {'数量':>8}")
        for label, cnt in dist["histogram"].items():
            print(f"    {label:>14}  {cnt:>8}")
    print()


def count_step3_text_lengths(root_dir: Path, bucket_edges: Optional[List[int]] = None) -> None:
    """
    扫描所有 *_step3_llm_cleaned.json，统计 text_ch 和 text_en 的字符长度、token 长度分布。
    """
    if bucket_edges is None:
        bucket_edges = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 10000]
    token_bucket_edges = [50, 100, 150, 200, 256, 384, 512, 768, 1024, 1536, 2048, 4096]

    files = discover_step3_llm_cleaned_files(root_dir)
    if not files:
        print(
            f"在 {root_dir} 下未找到任何 类别/语言目录/书名/{HYBRID_AUTO_DIR}/*{STEP3_LLM_CLEANED_SUFFIX} 文件。",
            file=sys.stderr,
        )
        sys.exit(1)

    lengths_ch: List[int] = []
    lengths_en: List[int] = []
    token_ch: List[int] = []
    token_en: List[int] = []

    token_counter = _get_token_counter()

    for input_path in tqdm(files, desc="扫描 JSON 文件", unit="个"):
        try:
            data = load_json(str(input_path))
        except Exception as e:
            print(f"[WARN] 读取失败 {input_path}：{e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            continue
        items = data.get("item") or data.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if "text_ch" in item:
                t = str(item.get("text_ch") or "")
                lengths_ch.append(len(t))
                if token_counter:
                    token_ch.append(token_counter(t))
            if "text_en" in item:
                t = str(item.get("text_en") or "")
                lengths_en.append(len(t))
                if token_counter:
                    token_en.append(token_counter(t))

    print(f"共扫描 {len(files)} 个 *_step3_llm_cleaned.json 文件。\n")

    # 字符长度分布
    _print_distribution("text_ch 字符长度分布", compute_length_distribution(lengths_ch, bucket_edges))
    _print_distribution("text_en 字符长度分布", compute_length_distribution(lengths_en, bucket_edges))

    # token 长度分布（需 tiktoken）
    if token_counter:
        _print_distribution(
            "text_ch token 长度分布 (tiktoken cl100k_base)",
            compute_length_distribution(token_ch, token_bucket_edges),
        )
        _print_distribution(
            "text_en token 长度分布 (tiktoken cl100k_base)",
            compute_length_distribution(token_en, token_bucket_edges),
        )
    else:
        print("[提示] 未安装 tiktoken，跳过 token 长度统计。可通过 pip install tiktoken 安装。\n")


def main():
    parser = argparse.ArgumentParser(
        description="统计 step3 LLM 清洗后 JSON 中 text_ch / text_en 字段的长度分布。"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="根文件夹目录。将扫描 root_dir/类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned.json。",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        print(f"根目录不存在或不是目录：{root}", file=sys.stderr)
        sys.exit(1)

    count_step3_text_lengths(root)


if __name__ == "__main__":
    main()
