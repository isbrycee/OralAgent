import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import argparse

try:
    import httpcore
except ImportError:
    httpcore = None  # type: ignore

# =========================
# 配置
# =========================

# JSON 解析失败时最多重新调用 LLM 的次数（不含首次），共最多调用 1 + 此值 次
MAX_PARSE_RETRIES = 2
# API 请求超时时间（秒），网络慢或接口慢时可适当调大
API_TIMEOUT = 300.0
# 连接/读取超时时的重试次数（每次间隔 2 秒）
API_TIMEOUT_RETRIES = 3

client = OpenAI(
    api_key="sk-",
    base_url="https://www.dmxapi.cn/v1/",
    timeout=API_TIMEOUT,
)

# =========================
# 语言检测：根据文件路径判断是中文版(_CH)还是英文版(_EN)
# =========================

def get_language_from_path(file_path: str) -> str:
    """
    若路径中包含 _CH，视为中文版（中→英）；
    若路径中包含 _EN，视为英文版（英→中）；
    否则默认视为中文版。
    """
    p = os.path.normpath(file_path)
    if "_EN" in p:
        return "en"
    if "_CH" in p:
        return "ch"
    return "ch"


# =========================
# LLM Prompt 设计
# =========================

# ---------- 中文版：输入为中文，输出 cleaned_text(中文) + english_text(英译) ----------
CLEANING_SYSTEM_PROMPT = """
## 口腔医学教材 OCR 文本清洗与知识提取 Prompt

你是一个 **「口腔医学教材 OCR 文本清洗与知识提取助手」**，目标是为构建高质量的【口腔医学专业知识库】提取可直接入库的医学知识条目。
你的任务包括 **三个步骤**（仅在内部执行，不要显示推理过程）：

### 一、任务步骤

1. **知识判定**：判断输入文本是否可作为一条有效的“口腔医学专业知识”收录到知识库。
2. **最小清洗**：如果可以，则在不改变医学意义的前提下去除无关部分并修正 OCR 错误；否则丢弃。  
3. **医学英文翻译**：若文本被保留（`keep = true`），需提供准确、专业、自然的英文医学翻译版本。

---

### 输出格式（必须严格遵守）

输出必须是一个合法的 JSON 对象：

```json
{
  "keep": true/false,
  "cleaned_text": "...",
  "english_text": "..."
}
```

要求：
- 当 `keep = false` 时，`cleaned_text` 与 `english_text` 必须为空。  
- 当 `keep = true` 时，`english_text` 必须是对 `cleaned_text` 的医学专业英文翻译。  
- 严禁输出多余字段、注释、数组或解释性文字。  
- 布尔值必须是原生布尔值（如 `true` 而非 `"true"`）。  

---

### 三、判定标准：能否作为“口腔医学知识库的知识条目”

【收录】（keep = true）
- 含有 **完整、明确、可独立表述的口腔医学知识**。  
- 叙述清晰，可作为事实性或机理性描述保存。

【不收录（keep = false）】
- 出版社、版权、编写、出版时间等信息。  
- 目录、页码、页眉、页脚、索引。  
- 纯标题或章节名：无正文时即使包含专业词汇（如“牙周炎的病因和发病机制”、“牙髓疾病的分类”），也删除。  
- 图表或图片说明、图注、表注。  
- 文献列表或参考文献。  
- 与口腔医学无关的政策性文字、广告、教学说明等。

---

### 四、清洗规则（仅在 `keep = true` 时执行）

执行以下最小化清洗，保持原意不变：

#### ① 删除纯指向图表的引用
例：
- “如图 3-1 所示，牙齿由牙冠和牙根组成。” → “牙齿由牙冠和牙根组成。”
- “见表 2-4。” → 删除整句（若无其他内容）。

#### ② 修正 OCR / 公式 / 单位错误
例：
- “Ca 2 + +” → “Ca2+”
- “mg / k g” → “mg/kg”
- “Na + +” → “Na+”
- “H 2 O 2” → “H2O2”

只修正常见符号、空格或上下标错误，不改变语义。

#### ③ 严格禁止的操作
- 不改写句子。
- 不总结或扩写。
- 不合并或拆分句子。
- 不替换专业术语。

---

### 五、非相关文本处理

如果文本判定为不可作为知识条目，或清洗后为空：
```json
{
  "keep": false,
  "cleaned_text": "",
  "english_text": ""
}
```

---

### 六、Few-shot 示例学习

你必须根据以下示例模式学习如何判断与输出。

---

#### 示例 1 — 有效知识（保留）

**输入文本：**
> 如图 3-1 所示，牙齿由牙冠和牙根组成。

**输出：**
```json
{
  "keep": true,
  "cleaned_text": "牙齿由牙冠和牙根组成。",
  "english_text": "The tooth consists of the crown and the root."
}
```

---

#### 示例 2 — 纯标题（删除）

**输入文本：**
> 第一节 牙周炎的病因和发病机制

**输出：**
```json
{
  "keep": false,
  "cleaned_text": "",
  "english_text": ""
}
```

（即使标题含医学词汇，也不视为知识条目。）

---

#### 示例 3 — 有效医学知识（保留）

**输入文本：**
> 牙周炎主要是由牙菌斑及其代谢产物引起牙周组织的慢性炎症性破坏。

**输出：**
```json
{
  "keep": true,
  "cleaned_text": "牙周炎主要是由牙菌斑及其代谢产物引起牙周组织的慢性炎症性破坏。",
  "english_text": "Periodontitis is primarily caused by dental plaque and its metabolic products, leading to chronic inflammatory destruction of periodontal tissues."
}
```

---

#### 示例 4 — 纯版权或出版信息（删除）

**输入文本：**
> 人民卫生出版社出版 2020 年第 2 版 版权所有

**输出：**
```json
{
  "keep": false,
  "cleaned_text": "",
  "english_text": ""
}
```

---

#### 示例 5 — 图表说明（删除）

**输入文本：**
> 图 5-2 颌骨骨化示意图 A. 骨化中心出现 B. 膜内骨化开始

**输出：**
```json
{
  "keep": false,
  "cleaned_text": "",
  "english_text": ""
}
```

---

#### 示例 6 — 有效临床知识（保留）

**输入文本：**
> 局部麻醉药通过阻断神经细胞膜上的钠离子通道，抑制动作电位的形成和传导，从而达到镇痛作用。

**输出：**
```json
{
  "keep": true,
  "cleaned_text": "局部麻醉药通过阻断神经细胞膜上的钠离子通道，抑制动作电位的形成和传导，从而达到镇痛作用。",
  "english_text": "Local anesthetics act by blocking sodium ion channels in the neuronal membrane, thereby inhibiting the generation and conduction of action potentials to achieve analgesia."
}
```

---

### 七、执行要求总结

- 严格按照 few-shot 行为输出 JSON 对象。  
- 仅保留能成为知识库条目的有效医学内容。  
- 清洗必须最小化且不改变句意。  
- 若保留，则输出精确、专业的英文译文。  
"""


# ---------- 英文版：输入为英文，输出 cleaned_text(英文) + chinese_text(中译) ----------
CLEANING_SYSTEM_PROMPT_EN = """
## Oral Medicine Textbook OCR Text Cleaning and Knowledge Extraction Prompt (English Source)

You are an **"Oral Medicine Textbook OCR Text Cleaning and Knowledge Extraction Assistant"** for building a high-quality **Oral Medicine Knowledge Base**. Your task has **three steps** (execute internally only; do not show reasoning):

### Step 1: Knowledge Judgment
Decide whether the input text is valid "oral medicine knowledge" that can be included in the knowledge base.

### Step 2: Minimal Cleaning
If yes, remove non-content parts and fix OCR errors without changing medical meaning; otherwise discard.

### Step 3: Medical Chinese Translation
If the text is kept (`keep = true`), provide an accurate, professional, natural **Chinese** medical translation.

---

### Output Format (strict JSON)

Output must be a single valid JSON object:

```json
{
  "keep": true/false,
  "cleaned_text": "...",
  "chinese_text": "..."
}
```

Rules:
- When `keep = false`, both `cleaned_text` and `chinese_text` must be empty strings.
- When `keep = true`, `chinese_text` must be a professional medical **Chinese** translation of `cleaned_text`.
- Do not output extra fields, comments, arrays, or explanatory text.
- Booleans must be native (e.g. `true` not `"true"`).

---

### Inclusion Criteria

**Include (keep = true):**
- Complete, clear, self-contained oral medicine knowledge.
- Clear narrative suitable as factual or mechanistic description.

**Exclude (keep = false):**
- Publisher, copyright, edition, publication date, etc.
- Table of contents, page numbers, headers, footers, index.
- Standalone headings or chapter titles with no body (even if they contain medical terms).
- Figure/table captions, legends.
- References or bibliography.
- Non–oral-medicine content (policy, ads, teaching instructions).

---

### Cleaning Rules (when keep = true)

- Remove figure/table references only (e.g. "As shown in Figure 3-1, ..." → "..."; "See Table 2-4." → delete if nothing else).
- Fix OCR / formula / unit errors (e.g. "Ca 2 + +" → "Ca2+", "mg / k g" → "mg/kg"). Do not change meaning.
- Do NOT rewrite, summarize, expand, merge, or split sentences; do NOT replace technical terms.

---

### Non-relevant Text

If the text is not valid knowledge or becomes empty after cleaning:
```json
{
  "keep": false,
  "cleaned_text": "",
  "chinese_text": ""
}
```

---

### Few-shot Examples (English source → Chinese translation)

**Example 1 — Valid knowledge (keep)**

Input:
> As shown in Figure 3-1, the tooth consists of the crown and the root.

Output:
```json
{
  "keep": true,
  "cleaned_text": "The tooth consists of the crown and the root.",
  "chinese_text": "牙齿由牙冠和牙根组成。"
}
```

**Example 2 — Standalone heading (discard)**

Input:
> Section 1 Etiology and Pathogenesis of Periodontitis

Output:
```json
{
  "keep": false,
  "cleaned_text": "",
  "chinese_text": ""
}
```

**Example 3 — Valid knowledge (keep)**

Input:
> Periodontitis is primarily caused by dental plaque and its metabolic products, leading to chronic inflammatory destruction of periodontal tissues.

Output:
```json
{
  "keep": true,
  "cleaned_text": "Periodontitis is primarily caused by dental plaque and its metabolic products, leading to chronic inflammatory destruction of periodontal tissues.",
  "chinese_text": "牙周炎主要是由牙菌斑及其代谢产物引起牙周组织的慢性炎症性破坏。"
}
```

**Example 4 — Copyright (discard)**

Input:
> Published by People's Medical Publishing House, 2nd edition 2020. All rights reserved.

Output:
```json
{
  "keep": false,
  "cleaned_text": "",
  "chinese_text": ""
}
```

---

### Summary

- Output only the JSON object as in the examples.
- Keep only valid oral medicine knowledge.
- Cleaning must be minimal and preserve meaning.
- When keeping, provide accurate professional **Chinese** translation in `chinese_text`.
"""


def build_user_prompt(text: str, file_name: str) -> str:
    return f"""
下面是一段从口腔医学教材 OCR 得到的【单独文本片段】（可能是行、句子或短段落），
请严格按照系统提示进行“相关性判断”和“最小清洗”。

请注意：
- 这段文本可能只是一页中的一小部分，你不能假设有额外上下文。
- 如果只看这段内容就足以判断为无关（如纯版权信息、纯目录行等），请直接 keep=false。
- 如果文本包含口腔医学专业知识，请按规则做最小清洗，不要总结，不要扩写。

文件名：{file_name}
原始文本（原样给出）：
{text}

请最终只输出一个 JSON 对象，格式为：
{{"keep": true/false, "cleaned_text": "...", "english_text": "..."}}。
"""


def build_user_prompt_en(text: str, file_name: str) -> str:
    """英文版 user prompt：输入为英文，要求输出 cleaned_text(英文) + chinese_text(中译)。"""
    return f"""
Below is a single text fragment from OCR of an oral medicine textbook (may be a line, sentence, or short paragraph).
Please follow the system prompt to judge relevance and perform minimal cleaning.

Notes:
- This fragment may be only a small part of a page; do not assume extra context.
- If the content is clearly irrelevant (e.g. copyright, table of contents), set keep=false.
- If it contains oral medicine knowledge, do minimal cleaning; do not summarize or expand.

File name: {file_name}
Original text:
{text}

Output exactly one JSON object in this form:
{{"keep": true/false, "cleaned_text": "...", "chinese_text": "..."}}.
"""


@dataclass
class CleanResult:
    keep: bool
    cleaned_text: str
    english_text: str  # 中→英时使用
    chinese_text: str  # 英→中时使用


def call_llm_clean(text: str, file_name: str, language: str = "ch") -> CleanResult:
    """
    调用 LLM 对单条文本进行：相关性判断 + 清洗。
    language: "ch" 为中文版（中→英），"en" 为英文版（英→中）。
    仅用 json.loads 解析；解析失败则重新调用 LLM，最多共 5 次，仍失败则跳过该条。
    """
    if not text or text.strip() == "":
        return CleanResult(keep=False, cleaned_text="", english_text="", chinese_text="")

    if language == "en":
        system_prompt = CLEANING_SYSTEM_PROMPT_EN
        user_content = build_user_prompt_en(text, file_name)
        translation_key = "chinese_text"
    else:
        system_prompt = CLEANING_SYSTEM_PROMPT
        user_content = build_user_prompt(text, file_name)
        translation_key = "english_text"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    max_attempts = 1 + MAX_PARSE_RETRIES  # 共 5 次

    for attempt in range(max_attempts):
        response = None
        for _ in range(API_TIMEOUT_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=8192,
                )
                break
            except Exception as e:
                is_timeout = (
                    (httpcore is not None and isinstance(e, (httpcore.ConnectTimeout, httpcore.ReadTimeout)))
                    or (type(e).__name__ in ("ConnectTimeout", "ReadTimeout", "TimeoutException"))
                )
                if is_timeout and _ < API_TIMEOUT_RETRIES - 1:
                    time.sleep(2)
                    continue
                raise
        if response is None:
            raise RuntimeError("API 请求未得到响应")
        content = response.choices[0].message.content.strip()
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            if attempt < max_attempts - 1:
                continue
            print(f"[WARN] {max_attempts} 次调用均 JSON 解析失败，跳过该条。最后输出：{content[:500]}...\n")
            return CleanResult(keep=False, cleaned_text="", english_text="", chinese_text="")
        if not isinstance(data, dict):
            if attempt < max_attempts - 1:
                continue
            print(f"[WARN] {max_attempts} 次调用均未得到合法 JSON 对象，跳过该条。最后输出：{content[:500]}...\n")
            return CleanResult(keep=False, cleaned_text="", english_text="", chinese_text="")
        keep = bool(data.get("keep", False))
        cleaned_text = data.get("cleaned_text", "")
        translation = data.get(translation_key, "")
        if keep and (not cleaned_text or cleaned_text.strip() == ""):
            keep = False
            cleaned_text = ""
        if language == "en":
            return CleanResult(
                keep=keep,
                cleaned_text=cleaned_text.strip(),
                english_text="",
                chinese_text=translation.strip(),
            )
        return CleanResult(
            keep=keep,
            cleaned_text=cleaned_text.strip(),
            english_text=translation.strip(),
            chinese_text="",
        )

    return CleanResult(keep=False, cleaned_text="", english_text="", chinese_text="")


# =========================
# 主处理逻辑
# =========================

def _get_text_key(item: Dict[str, Any]) -> Optional[str]:
    """返回 item 中用于正文的键：'text_ch' 或 'text_en'，若都不存在则返回 None。"""
    if "text_ch" in item:
        return "text_ch"
    if "text_en" in item:
        return "text_en"
    return None


def _should_process_item(item: Dict[str, Any]) -> bool:
    """仅对 type==paragraph 或 list_type==text_list 的项进行 LLM 处理。"""
    if item.get("type") == "paragraph":
        return True
    if item.get("list_type") == "text_list":
        return True
    return False


def _llm_task(args: Tuple[int, str, str, str]) -> Tuple[int, CleanResult]:
    """供线程池调用的单条 LLM 任务：(idx, text, file_name, language) -> (idx, result)。"""
    idx, text, file_name, language = args
    result = call_llm_clean(text, file_name, language=language)
    return (idx, result)


def process_single_file(input_path: str, output_path: str, concurrency: int = 5):
    """
    处理单个 *_step2_filter_rule_based.json 文件：
    - 根据路径中是否含 _CH / _EN 判断语言：_CH 为中文版（中→英），_EN 为英文版（英→中）
    - 加载原始 JSON（dict），读取其中的 item/items 列表
    - 仅对 type==paragraph 或 list_type==text_list 的项，按其 text_ch 或 text_en 调用 LLM（并行请求）
    - 仅保留 keep=True 的项，并替换对应 text_ch/text_en 为 cleaned_text，并写入翻译字段
    - 将结果写回为 dict，保留原顶层结构，仅替换 item/items 为清洗后的列表
    - concurrency: 同时发起的 LLM 请求数量。
    """
    file_name = os.path.basename(input_path)
    language = get_language_from_path(input_path)
    lang_label = "中文版(中→英)" if language == "ch" else "英文版(英→中)"
    print(f"Processing file: {input_path}  [{lang_label}]  concurrency={concurrency}")

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"[ERROR] 读取 JSON 失败：{input_path}，错误：{e}")
            return

    if not isinstance(data, dict):
        print(f"[WARN] 文件不是 dict 结构，跳过：{input_path}")
        return

    # 支持 "item" 或 "items" 作为列表字段名
    items = data.get("item") or data.get("items")
    if not isinstance(items, list):
        print(f"[WARN] 未找到 item/items 列表，跳过：{input_path}")
        return

    item_key = "item" if "item" in data else "items"

    # 第一遍：收集需要 LLM 处理的 (idx, text, file_name, language)
    work_list: List[Tuple[int, str, str, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if not _should_process_item(item):
            continue
        text_key = _get_text_key(item)
        if text_key is None:
            continue
        text = item.get(text_key, "")
        if not text or text.strip() == "":
            continue
        work_list.append((idx, text, file_name, language))

    # 并行执行 LLM 请求，带进度条
    results_by_idx: Dict[int, CleanResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_llm_task, task): task for task in work_list}
        with tqdm(total=len(work_list), desc="LLM 清洗", unit="条") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results_by_idx[idx] = result
                pbar.update(1)

    # 第二遍：按原始顺序遍历，组装 cleaned_items（非 LLM 项直接保留，LLM 项按结果保留或丢弃）
    cleaned_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if not _should_process_item(item):
            cleaned_items.append(dict(item))
            continue
        text_key = _get_text_key(item)
        if text_key is None:
            cleaned_items.append(dict(item))
            continue
        if idx not in results_by_idx:
            # 空文本等未提交 LLM 的，原样保留
            cleaned_items.append(dict(item))
            continue
        result = results_by_idx[idx]
        if result.keep:
            new_item = dict(item)
            new_item[text_key] = result.cleaned_text
            if language == "ch" and result.english_text:
                new_item["text_en"] = result.english_text
            elif language == "en" and result.chinese_text:
                new_item["text_ch"] = result.chinese_text
            cleaned_items.append(new_item)
        # keep=False 的项不加入 cleaned_items（丢弃）

    # 保留原 dict 的其余键，只替换 item/items
    out_data = dict(data)
    out_data[item_key] = cleaned_items

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"完成：{file_name}  原始条数 {len(items)}，LLM 处理 {len(work_list)} 条，保留 {len(cleaned_items)} 条。输出：{output_path}")


# 要查找的 step2 输出文件名后缀（与 step_2 输出一致）
STEP2_FILENAME_SUFFIX = "_step2_filter_rule_based.json"
# 路径中必须包含此目录名（即只处理 hybrid_auto 下的 step2 文件）
REQUIRED_DIR_IN_PATH = "hybrid_auto"


def discover_step2_json_files(root_dir: str) -> List[str]:
    """
    在根目录下递归查找所有 *_step2_filter_rule_based.json 文件。
    仅当路径中包含 hybrid_auto 时才纳入（目录结构示例：根目录/.../书名/hybrid_auto/书名_step2_filter_rule_based.json）。
    """
    root = os.path.abspath(root_dir)
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(STEP2_FILENAME_SUFFIX):
                full_path = os.path.join(dirpath, fname)
                if REQUIRED_DIR_IN_PATH in full_path:
                    found.append(full_path)
    return sorted(found)


def process_root(input_root: str, output_root: str, concurrency: int = 5):
    """
    输入为根目录，递归发现所有 *_step2_filter_rule_based.json，
    在输出根目录下保持相同相对路径，输出文件名为 xxx_step3_llm_cleaned.json。
    concurrency: 每个文件内同时发起的 LLM 请求数量。
    """
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    files = discover_step2_json_files(input_root)
    if not files:
        print(f"[WARN] 在 {input_root} 下未找到任何路径含 {REQUIRED_DIR_IN_PATH}/ 的 *{STEP2_FILENAME_SUFFIX} 文件。")
        return

    # 过滤掉已有 step3_llm_cleaned.json 的 step2 文件（已处理过则跳过）
    to_process = []
    for input_path in files:
        rel = os.path.relpath(input_path, input_root)
        dir_rel = os.path.dirname(rel)
        base_name = os.path.basename(input_path)
        stem = base_name[: -len(STEP2_FILENAME_SUFFIX)] if base_name.endswith(STEP2_FILENAME_SUFFIX) else base_name
        output_fname = stem + "_step3_llm_cleaned.json"
        output_path = os.path.join(output_root, dir_rel, output_fname)
        if os.path.isfile(output_path):
            print(f"[跳过] 已存在 {output_path}，跳过 step2 文件：{input_path}")
            continue
        to_process.append((input_path, output_path))

    print(f"共发现 {len(files)} 个 step2 文件，其中 {len(to_process)} 个待处理（已存在 step3 的已跳过），并行数 concurrency={concurrency}，开始处理。")
    for i, (input_path, output_path) in enumerate(to_process, 1):
        process_single_file(input_path, output_path, concurrency=concurrency)
        if i < len(to_process):
            print(f"[{i}/{len(to_process)}] 已处理。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 LLM 对口腔医学教材 step2 JSON 做相关性判断与清洗。输入为根目录，递归处理其下所有 *_step2_filter_rule_based.json。"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="/data/OralGPT/OralGPT-text-corpus-processed",
        help="输入根目录。将递归查找 子目录/.../hybrid_auto/*_step2_filter_rule_based.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/OralGPT/OralGPT-text-corpus-processed",
        help="输出根目录。输出保持与输入相同的相对路径，文件名为 *_step3_llm_cleaned.json",
    )
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="LLM 模型名称")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="同时发起的 LLM 请求数量（并行数），默认 5。可根据 API 限流情况调整。",
    )

    args = parser.parse_args()
    global MODEL_NAME
    MODEL_NAME = args.model

    process_root(args.input_root, args.output_root, concurrency=args.concurrency)
