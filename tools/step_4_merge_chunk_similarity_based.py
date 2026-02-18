import os
import json
import argparse
from typing import List, Dict
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import tiktoken

# -----------------------------
# 配置
# -----------------------------
END_TOKENS = set(["。", "？", "！", ".", "?", "!",]) #  ";", "：", ":", "；",
SIM_THRESHOLD = 0.8            # 语义相似度阈值
PAGE_DIFF_ALLOWED = 2           # 分句结束但相似也可合并的页码差
MAX_CHUNK_LEN = 512             # 每个输出 chunk 字数限制（按 token 数）
SENTENCE_END_REWARD = 0.2       # 是否是句子结尾的权重

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"   # 可改成 m3e-base, bge-large-zh 等
BATCH_SIZE = 64                   # 批量 encode 的 batch size

# 如果有 GPU，可改为 device="cuda"
model = SentenceTransformer(MODEL_NAME, device="cuda")  # , device="cuda"

# 使用 tiktoken cl100k_base（与 GPT-4/3.5 等一致）
encoding = tiktoken.get_encoding("cl100k_base")

# -----------------------------
# 工具函数
# -----------------------------
def clean_text(text: str) -> str:
    """
    尽量保留英文单词之间的空格，同时去除中文里被拆开的空格：
    “根 管 治 疗” -> “根管治疗”
    """
    # 先将各种空白统一为一个空格
    text = re.sub(r"\s+", " ", text)
    # 去掉连续中文之间的空格
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text.strip()


def is_end_of_sentence(text: str) -> bool:
    return len(text) > 0 and text[-1] in END_TOKENS


def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    余弦相似度。向量已做 L2 归一化的话，dot 即为 cos。
    """
    return float(np.dot(vec1, vec2))


def _tiktoken_offset_mapping(text: str, encoding: tiktoken.Encoding) -> List[tuple]:
    """根据 tiktoken 编码结果构建 (start, end) 字符偏移列表。"""
    token_ids = encoding.encode(text)
    offset_mapping = []
    start = 0
    for tid in token_ids:
        piece = encoding.decode([tid])
        end = start + len(piece)
        offset_mapping.append((start, end))
        start = end
    return offset_mapping


def chunk_text_by_tokens_with_sentence_boundary(text: str, max_token_len: int) -> List[str]:
    """
    将文本切分为多个 chunk，每个 chunk 的 token 数量不超过 max_token_len。
    切分时尽量在句子末尾（如句号、问号、感叹号）处分割。
    使用 tiktoken cl100k_base 计数。
    """
    chunks = []
    token_ids = encoding.encode(text)
    token_spans = _tiktoken_offset_mapping(text, encoding)
    n = len(token_ids)

    start = 0
    while start < n:
        end = start + max_token_len
        if end >= n:  # 如果已经到达文本末尾
            chunk_text = text[token_spans[start][0]:]
            chunks.append(chunk_text)
            break

        # 初始化分割位置为强制切分点
        split_pos = end - 1

        # 尝试在句子结束符处分割，向前查找
        for i in range(end - 1, start - 1, -1):
            if token_spans[i][1] > 0 and text[token_spans[i][1] - 1] in END_TOKENS:  # 检查句子结束符
                if i - start <= max_token_len:  # 确保长度限制
                    split_pos = i
                    break

        # 根据最终的分割位置切分文本
        chunk_text = text[token_spans[start][0]:token_spans[split_pos][1]]
        chunks.append(chunk_text)
        start = split_pos + 1  # 更新起始位置

    return chunks


def split_bilingual_segment_by_tokens(ch: str, en: str, max_token_len: int) -> List[tuple]:
    """
    当单段 (ch, en) 超过 max_token_len 时，按 token 数切分，保持中英对齐。
    返回 [(ch_part1, en_part1), (ch_part2, en_part2), ...]
    """
    ch_parts = chunk_text_by_tokens_with_sentence_boundary(ch, max_token_len)
    n = len(ch_parts)
    if n <= 1:
        return [(ch, en)] if ch.strip() or en.strip() else []

    en_token_ids = encoding.encode(en)
    total_en = len(en_token_ids)
    per_part = max(1, total_en // n)
    en_parts = []
    for i in range(n):
        start_i = i * per_part
        end_i = total_en if i == n - 1 else (i + 1) * per_part
        en_parts.append(encoding.decode(en_token_ids[start_i:end_i]))
    return list(zip(ch_parts, en_parts))

# -----------------------------
# 核心合并逻辑（单文件）
# -----------------------------
def process_json_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持 type 为 "paragraph" 或 "text"；读取 text_ch、text_en 与 page_index
    items: List[Dict] = []
    texts: List[str] = []

    for item in data.get("items", []):
        itype = item.get("type", "")
        if itype not in ("paragraph", "text"):
            continue
        raw_ch = item.get("text_ch", item.get("text", ""))
        raw_en = item.get("text_en", "")
        t_ch = clean_text(raw_ch)
        t_en = clean_text(raw_en)
        if not t_ch and not t_en:
            continue
        p = item.get("page_index", item.get("page_idx", 0))
        items.append({"text_ch": t_ch, "text_en": t_en, "page_index": p})
        # 用中文做 embedding，合并边界以中文语义为准
        texts.append(t_ch if t_ch else t_en)

    if not items:
        # 没有有效文本，直接输出空文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"items": []}, f, ensure_ascii=False, indent=2)
        return

    # 批量计算所有小段的 embedding（单位向量）
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # merged_chunks: 每个元素为 segment 列表 [(ch, en, page_index), ...]
    merged_chunks: List[List[tuple]] = []

    cur_segments: List[tuple] = []  # [(ch, en, page_index), ...]
    cur_text = ""   # 当前 chunk 的 text_ch 拼接，用于句子结束判断
    cur_page = None
    cur_emb_sum = None
    cur_count = 0

    for i in range(len(items)):
        t_ch = items[i]["text_ch"]
        t_en = items[i]["text_en"]
        p = items[i]["page_index"]
        t = t_ch if t_ch else t_en  # 用于 is_end_of_sentence
        new_emb = embeddings[i]

        if not cur_segments:
            cur_segments.append((t_ch, t_en, p))
            cur_text = t
            cur_page = p
            cur_emb_sum = new_emb.copy()
            cur_count = 1
            continue

        avg_emb = cur_emb_sum / (cur_count if cur_count > 0 else 1)
        norm = np.linalg.norm(avg_emb) + 1e-8
        avg_emb = avg_emb / norm

        sim = cosine_sim(avg_emb, new_emb)
        is_sentence_end = is_end_of_sentence(cur_text)
        weighted_sim = sim + (1.0 if not is_sentence_end else 0.0) * SENTENCE_END_REWARD

        if (not is_sentence_end and weighted_sim > SIM_THRESHOLD) or \
           (is_sentence_end and weighted_sim > SIM_THRESHOLD and abs(cur_page - p) <= PAGE_DIFF_ALLOWED):
            cur_segments.append((t_ch, t_en, p))
            cur_text += t
            cur_emb_sum += new_emb
            cur_count += 1
        else:
            merged_chunks.append(cur_segments)
            cur_segments = [(t_ch, t_en, p)]
            cur_text = t
            cur_page = p
            cur_emb_sum = new_emb.copy()
            cur_count = 1

    if cur_segments:
        merged_chunks.append(cur_segments)

    # 按 MAX_CHUNK_LEN 切分，保持 text_ch 与 text_en 一一对应
    final_chunks: List[Dict] = []
    for segments in merged_chunks:
        acc_ch: List[str] = []
        acc_en: List[str] = []
        acc_tokens = 0
        first_page = segments[0][2] if segments else 0

        for ch, en, page in segments:
            ch_tokens = len(encoding.encode(ch))
            if acc_tokens + ch_tokens > MAX_CHUNK_LEN and acc_ch:
                # 先输出当前累积
                final_chunks.append({
                    "type": "paragraph",
                    "text_ch": "".join(acc_ch),
                    "text_en": "".join(acc_en),
                    "page_index": first_page,
                })
                acc_ch, acc_en = [], []
                acc_tokens = 0
                first_page = page

            if ch_tokens > MAX_CHUNK_LEN:
                # 单段超长，按 token 切分并保持中英对齐
                sub_pairs = split_bilingual_segment_by_tokens(ch, en, MAX_CHUNK_LEN)
                for sub_ch, sub_en in sub_pairs:
                    if sub_ch.strip() or sub_en.strip():
                        final_chunks.append({
                            "type": "paragraph",
                            "text_ch": sub_ch,
                            "text_en": sub_en,
                            "page_index": page,
                        })
                acc_ch, acc_en = [], []
                acc_tokens = 0
            else:
                acc_ch.append(ch)
                acc_en.append(en)
                acc_tokens += ch_tokens

        if acc_ch or acc_en:
            final_chunks.append({
                "type": "paragraph",
                "text_ch": "".join(acc_ch),
                "text_en": "".join(acc_en),
                "page_index": first_page,
            })

    # 保存结果（与 step3 结构一致：顶层可带 summary，这里只输出 items 数组）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"items": final_chunks}, f, ensure_ascii=False, indent=2)


# -----------------------------
# 处理整个目录树（root_dir / 类别 / 语言目录 / 书名 / hybrid_auto / *_step3_llm_cleaned.json）
# 输出保存在与 step3 文件同一目录下，文件名后缀为 _step4_sim_chunk.json
# -----------------------------
STEP3_SUFFIX = "_step3_llm_cleaned.json"
STEP4_SUFFIX = "_step4_sim_chunk.json"


def collect_step3_files(root_dir: str) -> List[str]:
    """递归收集 root_dir 下所有 *_step3_llm_cleaned.json 的完整路径。"""
    result = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(STEP3_SUFFIX):
                result.append(os.path.join(dirpath, f))
    return sorted(result)


def process_folder(root_dir: str):
    """
    从 root_dir 开始遍历目录结构：
    root_dir / 类别 / 语言目录(如1_CH、1_EN) / 书名 / hybrid_auto / *_step3_llm_cleaned.json
    输出直接保存在对应的 step3 所在目录（如 hybrid_auto）下，文件名为 *_step4_sim_chunk.json
    """
    input_files = collect_step3_files(root_dir)
    if not input_files:
        print(f"No *{STEP3_SUFFIX} files found under {root_dir}")
        return

    for ipath in tqdm(input_files, desc="Processing files"):
        # 输出与输入同目录，仅将后缀改为 _step4_sim_chunk.json
        opath = ipath.replace(STEP3_SUFFIX, STEP4_SUFFIX)
        print(f"\nFile: {ipath}")
        process_json_file(ipath, opath)

    print("All files processed!")


# -----------------------------
# 入口（命令行运行）
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge OCR json texts into semantic chunks for RAG."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/OralGPT/OralGPT-text-corpus-demo/",
        help="根目录：将递归查找 类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned.json，输出同目录下 *_step4_sim_chunk.json"
    )

    args = parser.parse_args()
    process_folder(args.root_dir)