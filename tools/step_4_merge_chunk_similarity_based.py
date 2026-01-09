import os
import json
import argparse
from typing import List, Dict
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# -----------------------------
# 配置
# -----------------------------
END_TOKENS = set(["。", "？", "！", ".", "?", "!",]) #  ";", "：", ":", "；",
SIM_THRESHOLD = 0.8            # 语义相似度阈值
PAGE_DIFF_ALLOWED = 2           # 分句结束但相似也可合并的页码差
MAX_CHUNK_LEN = 512             # 每个输出 chunk 字数限制
SENTENCE_END_REWARD = 0.2       # 是否是句子结尾的权重

MODEL_NAME = "BAAI/bge-large-zh"   # 可改成 m3e-base, bge-large-zh 等
BATCH_SIZE = 64                   # 批量 encode 的 batch size

# 如果有 GPU，可改为 device="cuda"
model = SentenceTransformer(MODEL_NAME, device="cuda")  # , device="cuda"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 替换为你需要的模型

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


def chunk_text_by_tokens_with_sentence_boundary(text: str, max_token_len: int) -> List[str]:
    """
    将文本切分为多个 chunk，每个 chunk 的 token 数量不超过 max_token_len。
    切分时尽量在句子末尾（如句号、问号、感叹号）处分割。
    """
    chunks = []
    tokens = tokenizer.tokenize(text)  # 将文本分词
    token_spans = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]  # 获取 token 的字符位置
    token_ids = tokenizer.convert_tokens_to_ids(tokens)  # 转换为 token ID
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
            if text[token_spans[i][1] - 1] in END_TOKENS:  # 检查句子结束符
                if i - start <= max_token_len:  # 确保长度限制
                    split_pos = i
                    break

        # 根据最终的分割位置切分文本
        chunk_text = text[token_spans[start][0]:token_spans[split_pos][1]]
        chunks.append(chunk_text)
        start = split_pos + 1  # 更新起始位置

    return chunks

# -----------------------------
# 核心合并逻辑（单文件）
# -----------------------------
def process_json_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 先把有效的 text 抽出来并清洗，准备批量编码
    items: List[Dict] = []
    texts: List[str] = []

    for item in data:
        if item.get("type") != "text":
            continue
        raw_text = item.get("text", "")
        t = clean_text(raw_text)
        if not t:
            continue
        p = item.get("page_idx", 0)
        items.append({"text": t, "page_idx": p})
        texts.append(t)

    if not items:
        # 没有有效文本，直接输出空文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return

    # 批量计算所有小段的 embedding（单位向量）
    # show_progress_bar=True 会使用 tqdm 显示进度
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    merged_chunks: List[Dict] = []

    cur_text = ""
    cur_page = None
    cur_emb_sum = None  # 当前 chunk 内各小段 embedding 的和
    cur_count = 0       # 当前 chunk 内小段数量

    # 顺序遍历 items，embeddings[i] 与 items[i] 一一对应
    for i in range(len(items)):
        t = items[i]["text"]
        p = items[i]["page_idx"]
        new_emb = embeddings[i]

        # 第一段初始化
        if not cur_text:
            cur_text = t
            cur_page = p
            cur_emb_sum = new_emb.copy()
            cur_count = 1
            continue

        # 计算当前 chunk 的平均 embedding（再归一化一下）
        avg_emb = cur_emb_sum / (cur_count if cur_count > 0 else 1)
        norm = np.linalg.norm(avg_emb) + 1e-8
        avg_emb = avg_emb / norm

        sim = cosine_sim(avg_emb, new_emb)

        # 加权相似度计算
        is_sentence_end = is_end_of_sentence(cur_text)  # 返回 True 或 False
        weighted_sim = sim + (1.0 if not is_sentence_end else 0.0) * SENTENCE_END_REWARD

        print(f"Similarity: {sim:.4f}")
        print(f"Weighted similarity: {weighted_sim:.4f}")
        print(f"Current chunk: {cur_text}")
        print(f"New sentence: {t}")

        # 合并逻辑：
        # 1) 上一段没有自然结束标志 且 相似度高
        # 2) 上一段有结束标志 且 相似度高 且 页码差 <= PAGE_DIFF_ALLOWED
        if (not is_end_of_sentence(cur_text) and weighted_sim > SIM_THRESHOLD) or \
           (is_end_of_sentence(cur_text) and weighted_sim > SIM_THRESHOLD and abs(cur_page - p) <= PAGE_DIFF_ALLOWED):
            # 合并
            cur_text += t
            # page_idx 保持为 chunk 首段的页码
            cur_emb_sum += new_emb
            cur_count += 1
        else:
            # 先把当前 chunk 收尾
            merged_chunks.append({"text": cur_text, "page_idx": cur_page})
            # 开启新 chunk
            cur_text = t
            cur_page = p
            cur_emb_sum = new_emb.copy()
            cur_count = 1

    # 收尾
    if cur_text:
        merged_chunks.append({"text": cur_text, "page_idx": cur_page})

    # 第二次处理：控制每个 chunk 字数不超过 MAX_CHUNK_LEN
    final_chunks: List[Dict] = []
    for chunk in merged_chunks:
        parts = chunk_text_by_tokens_with_sentence_boundary(chunk["text"], MAX_CHUNK_LEN)
        for part in parts:
            if part.strip():
                final_chunks.append({"type": "text", "text": part, "page_idx": chunk["page_idx"]})

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)


# -----------------------------
# 处理整个文件夹
# -----------------------------
def process_folder(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.endswith("_llm_cleaned.json")
    ]
    # 文件级进度条
    for filename in tqdm(files, desc="Processing files"):
        ipath = os.path.join(input_dir, filename)
        basename, _ = os.path.splitext(filename)
        opath = os.path.join(output_dir, f"{basename}_chunk_merged.json")

        print(f"\nFile: {filename}")  # 与 encode 内部的进度条一起用时更清晰
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
        "--input_dir",
        type=str,
        default="/home/jinghao/projects/OralGPT-Agent/Corpus/step_3_llm_refine/",
        help="输入 json 文件夹（含 *_filter_rule_base.json）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jinghao/projects/OralGPT-Agent/Corpus/step_4_sim_chunk/",
        help="输出合并后的 json 文件夹"
    )

    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)