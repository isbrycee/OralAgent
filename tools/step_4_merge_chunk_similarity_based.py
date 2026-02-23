import os
import json
import argparse
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import tiktoken

# -----------------------------
# 配置
# -----------------------------
END_TOKENS = set(["。", "？", "！", ".", "?", "!"])
SECONDARY_BOUNDARY = set(["，", "、", ",", "；", ";", "：", ":"])  # 次要切分点，无句号时使用
SIM_THRESHOLD = 0.8            # 语义相似度阈值
PAGE_DIFF_ALLOWED = 2           # 分句结束但相似也可合并的页码差
MAX_CHUNK_LEN = 768             # 每个输出 chunk 字数限制（按 token 数）
SENTENCE_END_REWARD = 0.1       # 句子未结束时合并的 sim 奖励（避免阈值过宽导致误合并）
SEGMENT_JOIN = "\n"             # 合并段落时用的分隔符

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"   # 可改成 m3e-base, bge-large-zh 等
BATCH_SIZE = 32                   # 批量 encode 的 batch size

# 如果有 GPU，可改为 device="cuda"
model = SentenceTransformer(MODEL_NAME, device="cuda")  # , device="cuda"

# 使用 tiktoken cl100k_base（与 GPT-4/3.5 等一致）
encoding = tiktoken.get_encoding("cl100k_base")

# -----------------------------
# 工具函数
# -----------------------------
def _removed_clean_text(text: str) -> str:
    """
    尽量保留英文单词之间的空格，同时去除中文里被拆开的空格：
    “根 管 治 疗” -> “根管治疗”
    """
    # 先将各种空白统一为一个空格
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


def _safe_char_at(text: str, idx: int) -> str:
    """安全获取 text[idx]，越界返回空字符。"""
    if 0 <= idx < len(text):
        return text[idx]
    return ""


def chunk_text_by_tokens_with_sentence_boundary(text: str, max_token_len: int) -> List[str]:
    """
    将文本切分为多个 chunk，每个 chunk 的 token 数量不超过 max_token_len。
    切分时尽量在句子末尾（如句号、问号、感叹号）处分割；若无则尝试逗号、分号等。
    使用 tiktoken cl100k_base 计数。
    """
    chunks = []
    text_len = len(text)
    token_ids = encoding.encode(text)
    token_spans = _tiktoken_offset_mapping(text, encoding)
    n = len(token_ids)

    start = 0
    while start < n:
        end = start + max_token_len
        if end >= n:  # 如果已经到达文本末尾
            s0 = min(token_spans[start][0], text_len)
            chunk_text = text[s0:]
            chunks.append(chunk_text)
            break

        # 初始化分割位置为强制切分点
        split_pos = end - 1

        # 1) 优先在句子结束符处分割
        for i in range(end - 1, start - 1, -1):
            end_char_idx = token_spans[i][1] - 1
            if end_char_idx >= 0 and end_char_idx < text_len and _safe_char_at(text, end_char_idx) in END_TOKENS:
                if i - start <= max_token_len:
                    split_pos = i
                    break
        else:
            # 2) 无句号时，尝试在逗号、分号等次要边界分割
            for i in range(end - 1, start - 1, -1):
                end_char_idx = token_spans[i][1] - 1
                if end_char_idx >= 0 and end_char_idx < text_len and _safe_char_at(text, end_char_idx) in SECONDARY_BOUNDARY:
                    if i - start <= max_token_len:
                        split_pos = i
                        break

        # 切片时做越界防护
        s0 = min(max(0, token_spans[start][0]), text_len)
        s1 = min(max(0, token_spans[split_pos][1]), text_len)
        chunk_text = text[s0:s1]
        chunks.append(chunk_text)
        start = split_pos + 1

    return chunks


def split_bilingual_segment_by_tokens(ch: str, en: str, max_token_len: int) -> List[tuple]:
    """
    当单段 (ch, en) 超过 max_token_len 时，按 token 数切分，保持中英对齐。
    按中英 token 比例分配英文切分点，使各部分 token 量更均衡。
    返回 [(ch_part1, en_part1), (ch_part2, en_part2), ...]
    """
    if not (ch.strip() or en.strip()):
        return []

    ch_parts = chunk_text_by_tokens_with_sentence_boundary(ch, max_token_len)
    n = len(ch_parts)
    if n <= 1:
        return [(ch, en)]

    # 按 token 比例分配英文：每段中文的 token 占比决定对应英文的 token 数
    en_token_ids = encoding.encode(en)
    total_en = len(en_token_ids)
    if total_en == 0:
        return list(zip(ch_parts, [""] * n))

    # 计算每段中文的 token 边界（累计）
    ch_boundaries = [0]
    for part in ch_parts:
        ch_boundaries.append(ch_boundaries[-1] + len(encoding.encode(part)))
    total_ch = ch_boundaries[-1]
    if total_ch == 0:
        # 中文无 token 时退化为等分英文
        per_part = max(1, total_en // n)
        en_parts = []
        for i in range(n):
            start_i = i * per_part
            end_i = total_en if i == n - 1 else (i + 1) * per_part
            en_parts.append(encoding.decode(en_token_ids[start_i:end_i]))
        return list(zip(ch_parts, en_parts))

    # 按比例映射到英文 token 边界
    en_parts = []
    for i in range(n):
        ratio_start = ch_boundaries[i] / total_ch
        ratio_end = ch_boundaries[i + 1] / total_ch
        start_i = min(int(ratio_start * total_en), total_en)
        end_i = total_en if i == n - 1 else min(int(ratio_end * total_en), total_en)
        end_i = max(end_i, start_i + 1) if i < n - 1 and end_i <= start_i else end_i
        en_parts.append(encoding.decode(en_token_ids[start_i:end_i]))
    return list(zip(ch_parts, en_parts))

# -----------------------------
# 核心合并逻辑（单文件）
# -----------------------------
def process_json_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 保留原始 meta 字段到最终输出
    meta = data.get("meta")

    # 支持 type 为 "paragraph" 或 "text"；读取 text_ch、text_en 与 page_number
    items: List[Dict] = []
    texts: List[str] = []

    # 用于打印示例：各类处理的 (原始, 处理后)，每类最多几条
    merge_examples: List[tuple] = []       # (合并前 segments [(ch,en),...], 合并后 chunk)
    long_split_examples: List[tuple] = []  # (原始 ch, en, 切分后的 [(sub_ch, sub_en), ...])

    for item in data.get("items", []):
        itype = item.get("type", "")
        if itype not in ("paragraph", "text"):
            continue
        raw_ch = item.get("text_ch", item.get("text", ""))
        raw_en = item.get("text_en", "")
        t_ch = raw_ch.strip()
        t_en = raw_en.strip()
        if not t_ch and not t_en:
            continue
        p = item.get("page_number", item.get("page_idx", 0))
        items.append({"text_ch": t_ch, "text_en": t_en, "page_number": p})
        # 用中文做 embedding，合并边界以中文语义为准
        texts.append(t_ch if t_ch else t_en)

    count_before = len(items)  # 处理前 text 数量

    if not items:
        # 没有有效文本，直接输出空文件
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        output_obj = {"items": []}
        if meta is not None:
            output_obj["meta"] = meta
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, ensure_ascii=False, indent=2)
        print(f"  [空文件] 处理前 text 数量: 0, 处理后 text 数量: 0")
        return

    # 批量计算所有小段的 embedding（单位向量）
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # merged_chunks: 每个元素为 segment 列表 [(ch, en, page_number), ...]
    merged_chunks: List[List[tuple]] = []

    cur_segments: List[tuple] = []  # [(ch, en, page_number), ...]
    cur_text = ""   # 当前 chunk 的 text_ch 拼接，用于句子结束判断
    cur_page = None
    cur_emb_sum = None
    cur_count = 0

    for i in range(len(items)):
        t_ch = items[i]["text_ch"]
        t_en = items[i]["text_en"]
        p = items[i]["page_number"]
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
            cur_text += (SEGMENT_JOIN if cur_text else "") + t
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

    # 按 MAX_CHUNK_LEN 切分，保持 text_ch 与 text_en 一一对应；记录每条来自几个 segment 及合并前原文（用于打印示例）
    final_chunks: List[Dict] = []
    for segments in merged_chunks:
        acc_ch: List[str] = []
        acc_en: List[str] = []
        acc_tokens = 0
        first_page = segments[0][2] if segments else 0
        seg_start_idx = 0  # 当前 acc 对应的 segments 起始下标

        for seg_idx, (ch, en, page) in enumerate(segments):
            ch_tokens = len(encoding.encode(ch))
            if acc_tokens + ch_tokens > MAX_CHUNK_LEN and acc_ch:
                # 先输出当前累积
                n_seg = len(acc_ch)
                pre_segs = [(segments[i][0], segments[i][1]) for i in range(seg_start_idx, seg_start_idx + n_seg)]
                final_chunks.append({
                    "type": "paragraph",
                    "text_ch": "".join(acc_ch),
                    "text_en": "".join(acc_en),
                    "page_number": first_page,
                    "_n_segments": n_seg,
                    "_pre_segments": pre_segs,
                })
                acc_ch, acc_en = [], []
                acc_tokens = 0
                first_page = page
                seg_start_idx = seg_idx

            if ch_tokens > MAX_CHUNK_LEN:
                # 单段超长，按 token 切分并保持中英对齐
                sub_pairs = split_bilingual_segment_by_tokens(ch, en, MAX_CHUNK_LEN)
                if len(long_split_examples) < 3:
                    long_split_examples.append((ch, en, sub_pairs))
                for sub_ch, sub_en in sub_pairs:
                    if sub_ch.strip() or sub_en.strip():
                        final_chunks.append({
                            "type": "paragraph",
                            "text_ch": sub_ch,
                            "text_en": sub_en,
                            "page_number": page,
                            "_n_segments": 1,
                            "_pre_segments": [(ch, en)],
                        })
                acc_ch, acc_en = [], []
                acc_tokens = 0
                seg_start_idx = seg_idx + 1
            else:
                acc_ch.append(ch)
                acc_en.append(en)
                acc_tokens += ch_tokens

        if acc_ch or acc_en:
            n_seg = len(acc_ch)
            pre_segs = [(segments[i][0], segments[i][1]) for i in range(seg_start_idx, seg_start_idx + n_seg)]
            final_chunks.append({
                "type": "paragraph",
                "text_ch": "".join(acc_ch),
                "text_en": "".join(acc_en),
                "page_number": first_page,
                "_n_segments": n_seg,
                "_pre_segments": pre_segs,
            })

    # 收集“多句合并”示例（最多 3 条）
    merged_only = [c for c in final_chunks if c.get("_n_segments", 0) >= 2]
    for c in merged_only:
        if len(merge_examples) >= 3:
            break
        merge_examples.append((c.get("_pre_segments", []), (c.get("text_ch", ""), c.get("text_en", ""))))

    # ---------- 统一打印：所有处理类型 + 原始 vs 处理后示例（每类最多几条）----------
    def _preview(s: str, max_len: int = 100) -> str:
        s = (s[:max_len] + "…") if len(s) > max_len else s
        return s.replace("\n", " ")

    print("\n  ========== 处理情况与示例（每类仅展示几条）==========")

    if merge_examples:
        print(f"\n  【1】多句合并 (共 {len(merged_only)} 处，展示 {len(merge_examples)} 条)")
        for idx, (pre_segs, (out_ch, out_en)) in enumerate(merge_examples, 1):
            print(f"      示例 {idx}:")
            print(f"        原始（{len(pre_segs)} 段）:")
            for i, (a, b) in enumerate(pre_segs[:5], 1):
                print(f"          段{i} CH: {_preview(a)}")
                if b:
                    print(f"          段{i} EN: {_preview(b)}")
            print(f"        合并后 1 条:")
            print(f"          CH: {_preview(out_ch)}")
            if out_en:
                print(f"          EN: {_preview(out_en)}")
    else:
        print("\n  【1】多句合并: 无（未出现多段合并成一条）")

    if long_split_examples:
        print(f"\n  【2】单段超长按 token 切分 (共 {len(long_split_examples)} 条示例)")
        for idx, (orig_ch, orig_en, sub_pairs) in enumerate(long_split_examples[:3], 1):
            print(f"      示例 {idx}:")
            print(f"        原始 CH: {_preview(orig_ch, 120)}")
            if orig_en:
                print(f"        原始 EN: {_preview(orig_en, 120)}")
            print(f"        切分后（{len(sub_pairs)} 段）:")
            for i, (sc, se) in enumerate(sub_pairs[:3], 1):
                print(f"          段{i} CH: {_preview(sc)}")
                if se:
                    print(f"          段{i} EN: {_preview(se)}")
            if len(sub_pairs) > 3:
                print(f"          ... 共 {len(sub_pairs)} 段")
    else:
        print("\n  【2】单段超长切分: 无")

    print(f"\n  >>> 处理前 text 数量: {count_before}, 处理后 text 数量: {len(final_chunks)}")
    print(f"  ---------- 最终条数: {len(final_chunks)} ----------\n")

    # 写入 JSON 前去掉临时字段
    for c in final_chunks:
        c.pop("_n_segments", None)
        c.pop("_pre_segments", None)

    # 保存结果（保留顶层 meta，与 step3 结构一致）
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output_obj = {"items": final_chunks}
    if meta is not None:
        output_obj["meta"] = meta
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# 处理整个目录树（root_dir / 类别 / 语言目录 / 书名 / hybrid_auto / *_step3_llm_cleaned_w_meta.json）
# 输出保存在与 step3 文件同一目录下，文件名后缀为 _step4_sim_chunk.json
# -----------------------------
STEP3_SUFFIX = "_step3_llm_cleaned_w_meta.json"
STEP4_SUFFIX = "_step4_sim_chunk.json"


def collect_step3_files(root_dir: str) -> List[str]:
    """递归收集 root_dir 下所有 *_step3_llm_cleaned_w_meta.json 的完整路径。"""
    result = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(STEP3_SUFFIX):
                result.append(os.path.join(dirpath, f))
    return sorted(result)


def process_folder(root_dir: str):
    """
    从 root_dir 开始遍历目录结构：
    root_dir / 类别 / 语言目录(如1_CH、1_EN) / 书名 / hybrid_auto / *_step3_llm_cleaned_w_meta.json
    输出直接保存在对应的 step3 所在目录（如 hybrid_auto）下，文件名为 *_step4_sim_chunk.json
    """
    input_files = collect_step3_files(root_dir)
    if not input_files:
        print(f"No *{STEP3_SUFFIX} files found under {root_dir}")
        return

    failed = []
    for ipath in tqdm(input_files, desc="Processing files"):
        opath = ipath.replace(STEP3_SUFFIX, STEP4_SUFFIX)
        if os.path.isfile(opath):
            print(f"\nSkip (already processed): {ipath}")
            continue
        print(f"\nFile: {ipath}")
        try:
            process_json_file(ipath, opath)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(ipath)
    if failed:
        print(f"\nFailed {len(failed)} file(s):")
        for f in failed:
            print(f"  - {f}")

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
        default="/data/OralGPT/OralGPT-text-corpus-processed/",
        help="根目录：将递归查找 类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned_w_meta.json，输出同目录下 *_step4_sim_chunk.json"
    )

    args = parser.parse_args()
    process_folder(args.root_dir)