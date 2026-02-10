import argparse
import json
import sys
import re
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional

# 每个数据集下要处理的子目录与文件后缀
HYBRID_AUTO_DIR = "hybrid_auto"
CONTENT_LIST_V2_SUFFIX = "_content_list_v2.json"
STEP2_OUTPUT_SUFFIX = "_step2_filter_rule_based.json"

# 句子结尾符号（中英文）
SENTENCE_END_CHARS = set("。.!?！？;；…")
# 以这些符号结尾视为未结束，必须与下一段合并（顿号、全角逗号等）
INCOMPLETE_END_CHARS = set("、，,")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_page_list(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], list))


def extract_page_number(page_items: List[Dict[str, Any]]) -> Optional[int]:
    candidates = []
    for el in page_items:
        if el.get("type") == "page_number":
            content = el.get("content", {})
            nodes = content.get("page_number_content", [])
            text_parts = []
            for n in nodes:
                if isinstance(n, dict) and n.get("type") == "text":
                    text_parts.append(n.get("content", ""))
            text = "".join(text_parts).strip()
            if text:
                m = re.search(r"\d+", text)
                if m:
                    try:
                        candidates.append(int(m.group(0)))
                    except ValueError:
                        pass
    if candidates:
        return candidates[0]
    return None


def extract_text_nodes(nodes: List[Dict[str, Any]]) -> str:
    parts = []
    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        t = n.get("type")
        c = n.get("content", "")
        if t in ("text", "equation_inline"):
            parts.append(str(c))
        else:
            # 可扩展处理其他类型，如 equation_display 等
            pass
    return "".join(parts).strip()


def extract_image_caption(nodes: List[Dict[str, Any]]) -> str:
    return extract_text_nodes(nodes)


def extract_table_caption(nodes: List[Dict[str, Any]]) -> str:
    return extract_text_nodes(nodes)


def has_sentence_end(text: str) -> bool:
    """检查文本是否以句子结尾符号结束（视为完整句子）。
    以顿号、全角逗号等结尾视为未结束，返回 False，以便与下一段合并。
    """
    t = (text or "").strip()
    if len(t) == 0:
        return False
    # 以顿号、全角逗号结尾的视为未结束，需要与下一段合并
    if t[-1] in INCOMPLETE_END_CHARS:
        return False
    return t[-1] in SENTENCE_END_CHARS


def filter_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去掉 page_number 为 null、type 为 image、type 为 table 的元素，保持原有顺序。"""
    return [
        el for el in items
        if el.get("page_number") is not None
        and el.get("type") != "image"
        and el.get("type") != "table"
    ]


def merge_incomplete_paragraphs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对每个 type 为 paragraph 的项检查 text 是否以句子结尾符号结束；
    若不是，则与下一个紧邻的 paragraph 合并，并递归检查合并后的文本。
    合并时只更新 text、text_length；bbox、page_number、page_index、element_index_in_page
    均保留第一个被合并元素的值（即跨页合并时 page_number 取第一段所在页）。
    保持列表中非 paragraph 元素的顺序不变。
    """
    result: List[Dict[str, Any]] = []
    i = 0
    while i < len(items):
        el = items[i]
        if el.get("type") != "paragraph":
            result.append(el)
            i += 1
            continue
        # 从当前 paragraph 开始，不断与下一个 paragraph 合并直到成为完整句子或没有下一个 paragraph
        current = dict(el)
        j = i + 1
        while not has_sentence_end(current.get("text", "")) and j < len(items) and items[j].get("type") == "paragraph":
            next_el = items[j]
            next_text = next_el.get("text") or ""
            merged_text = (current.get("text") or "").strip()
            if merged_text and next_text.strip():
                merged_text = merged_text + " " + next_text.strip()
            else:
                merged_text = (merged_text + next_text).strip() or (next_text or "").strip()
            current["text"] = merged_text
            current["text_length"] = len(merged_text)
            # bbox 只保留第一个被合并元素的；page_number 已是第一个元素的，不随 next 改变
            j += 1
        # 保持与原始 paragraph 相同的键顺序
        ordered = OrderedDict([
            ("page_index", current["page_index"]),
            ("page_number", current["page_number"]),
            ("type", current["type"]),
            ("bbox", current["bbox"]),
            ("element_index_in_page", current["element_index_in_page"]),
            ("text", current["text"]),
            ("text_length", current["text_length"]),
        ])
        result.append(dict(ordered))
        i = j
    return result


def summarize_structure(pages: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["total_pages"] = len(pages)

    per_page_counts = []
    type_counter = Counter()
    page_numbers = []

    for page in pages:
        per_page_counts.append(len(page))
        local_counter = Counter([el.get("type") for el in page])
        type_counter.update(local_counter)
        pn = extract_page_number(page)
        page_numbers.append(pn)

    summary["elements_per_page"] = per_page_counts
    summary["type_counts_global"] = dict(type_counter)
    summary["page_numbers"] = page_numbers

    recognized = [pn for pn in page_numbers if pn is not None]
    if recognized:
        continuous = all(
            (recognized[i] + 1 == recognized[i + 1]) for i in range(len(recognized) - 1)
        )
        summary["page_numbers_continuous"] = continuous
    else:
        summary["page_numbers_continuous"] = None

    return summary


def extract_list_items_texts(list_items: List[Dict[str, Any]]) -> List[str]:
    """
    从 list.content.list_items 中抽取每条目的文本。
    每个条目的 item_content 是一个节点列表，支持 text/equation_inline。
    """
    texts: List[str] = []
    for item in list_items or []:
        nodes = item.get("item_content", [])
        t = extract_text_nodes(nodes)
        if t is None:
            t = ""
        texts.append(t)
    return texts


def build_item_index(
    pages: List[List[Dict[str, Any]]],
    flatten_lists: bool = False
) -> List[Dict[str, Any]]:
    """
    输出每个 image、table、paragraph、list 的记录，附带所在页码、bbox等。
    如果 flatten_lists=True，则额外把 list 的每个条目打平为独立的 list_item 记录。
    """
    items = []
    last_known_page_number: Optional[int] = None

    for page_idx, page in enumerate(pages):
        page_number = extract_page_number(page)
        if page_number is not None:
            last_known_page_number = page_number
        else:
            if last_known_page_number is not None:
                last_known_page_number = last_known_page_number + 1

        for el_idx, el in enumerate(page):
            t = el.get("type")
            if t not in ("image", "table", "paragraph", "list"):
                continue

            base: Dict[str, Any] = {
                "page_index": page_idx,
                "page_number": last_known_page_number,
                "type": t,
                "bbox": el.get("bbox"),
                "element_index_in_page": el_idx,
            }

            if t == "image":
                c = el.get("content", {})
                src = c.get("image_source", {})
                rec = dict(base)
                rec["image_path"] = src.get("path")
                rec["caption"] = extract_image_caption(c.get("image_caption", []))
                rec["footnote_count"] = len(c.get("image_footnote", []))
                items.append(rec)

            elif t == "table":
                c = el.get("content", {})
                src = c.get("image_source", {})
                rec = dict(base)
                rec["table_image_path"] = src.get("path")
                rec["caption"] = extract_table_caption(c.get("table_caption", []))
                rec["table_type"] = c.get("table_type")
                rec["table_nest_level"] = c.get("table_nest_level")
                rec["html"] = c.get("html")
                rec["footnote_count"] = len(c.get("table_footnote", []))
                items.append(rec)

            elif t == "paragraph":
                c = el.get("content", {})
                nodes = c.get("paragraph_content", [])
                text = extract_text_nodes(nodes)
                rec = dict(base)
                rec["text"] = (text or "")
                rec["text_length"] = len(text or "")
                items.append(rec)

            elif t == "list":
                c = el.get("content", {})
                rec = dict(base)
                rec["list_type"] = c.get("list_type")
                list_items = c.get("list_items", [])
                texts = extract_list_items_texts(list_items)
                joined = "\n".join(texts)
                rec["items_count"] = len(texts)
                rec["items_texts"] = texts  # 保留每条的完整文本
                rec["text"] = joined
                rec["text_length"] = len(joined)
                items.append(rec)

                if flatten_lists:
                    # 为每个 list 条目生成单独记录
                    for i, txt in enumerate(texts):
                        li_rec: Dict[str, Any] = {
                            "page_index": page_idx,
                            "page_number": last_known_page_number,
                            "type": "list_item",
                            "bbox": el.get("bbox"),
                            "element_index_in_page": el_idx,      # 父 list 在页中的索引
                            "item_index_in_list": i,              # 当前条目在 list 中的索引
                            "list_type": rec["list_type"],
                            "text": (txt or ""),
                            "text_length": len(txt or ""),
                        }
                        items.append(li_rec)

    return items


def discover_content_list_v2_files(root_dir: Path) -> List[Path]:
    """
    在根目录下查找所有 hybrid_auto/*_content_list_v2.json 文件。
    目录结构：root_dir / 类别 / 语言目录(如1_CH、1_EN) / 书名 / hybrid_auto / *_content_list_v2.json
    例如：OralGPT-text-corpus-processed/1-口腔解剖生理学.../1_CH/口腔解剖生理学（人卫...）/hybrid_auto/xxx_content_list_v2.json
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
                    if f.is_file() and f.name.endswith(CONTENT_LIST_V2_SUFFIX):
                        found.append(f)
    return sorted(found)


def process_one_file(
    input_path: Path,
    out_path: Path,
    flatten_lists: bool = False,
    print_items: bool = False,
) -> None:
    """对单个 _content_list_v2.json 做过滤并写入 step2 输出。"""
    data = load_json(str(input_path))
    if not is_page_list(data):
        raise ValueError(f"输入 JSON 顶层结构不是 List[List[dict]]（即页列表）：{input_path}")

    pages: List[List[Dict[str, Any]]] = data
    summary = summarize_structure(pages)
    items = build_item_index(pages, flatten_lists=flatten_lists)
    items = filter_items(items)
    items = merge_incomplete_paragraphs(items)

    if print_items:
        print(f"====== 条目索引 {input_path.name} ======")
        for r in items:
            t = r["type"]
            if t == "image":
                print(f"[Image] p#{r['page_number']} idx{r['page_index']} bbox={r['bbox']} path={r.get('image_path')} caption={r.get('caption')}")
            elif t == "table":
                print(f"[Table] p#{r['page_number']} idx{r['page_index']} bbox={r['bbox']} caption={r.get('caption')} table_type={r.get('table_type')}")
            elif t == "paragraph":
                preview = (r.get("text") or "").replace("\n", " ")
                print(f"[Para ] p#{r['page_number']} idx{r['page_index']} bbox={r['bbox']} text={preview}")
            elif t == "list":
                preview = (r.get("text") or "").replace("\n", " ")
                print(f"[List ] p#{r['page_number']} idx{r['page_index']} bbox={r['bbox']} list_type={r.get('list_type')} items_count={r.get('items_count')} preview={preview}")
            elif t == "list_item":
                preview = (r.get("text") or "").replace("\n", " ")
                print(f"[LItem] p#{r['page_number']} idx{r['page_index']} parent_el_idx={r['element_index_in_page']} item_idx={r['item_index_in_list']} list_type={r.get('list_type')} preview={preview}")

    input_str = str(input_path)
    if "_CH" in input_str:
        text_key = "text_ch"
    elif "_EN" in input_str:
        text_key = "text_en"
    else:
        text_key = "text"

    output_items = []
    for it in items:
        out_it = OrderedDict()
        for k, v in it.items():
            if k == "text":
                out_it[text_key] = v
            elif k == "type":
                out_it["type"] = "paragraph" if v == "list" else v
            else:
                out_it[k] = v
        output_items.append(out_it)

    out_data = OrderedDict([
        ("summary", summary),
        ("items", output_items),
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="分析教学资料 JSON 结构：输入为根目录，处理其下各数据集 hybrid_auto/*_content_list_v2.json。"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="根文件夹目录（如 OralGPT-text-corpus-processed）。将扫描 root_dir/类别/语言目录(如1_CH,1_EN)/书名/hybrid_auto/*_content_list_v2.json 并逐个处理。",
    )
    parser.add_argument("--print-items", action="store_true", help="在控制台打印各项索引记录")
    parser.add_argument("--flatten-lists", action="store_true", help="将列表条目打平为独立记录（list_item）")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        print(f"根目录不存在或不是目录：{root}", file=sys.stderr)
        sys.exit(1)

    files = discover_content_list_v2_files(root)
    if not files:
        print(
            f"在 {root} 下未找到任何 类别/语言目录/书名/{HYBRID_AUTO_DIR}/*{CONTENT_LIST_V2_SUFFIX} 文件。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"共发现 {len(files)} 个 *_content_list_v2.json 文件，开始处理。")

    for i, input_path in enumerate(files, 1):
        # 输出与输入同目录，文件名由 _content_list_v2.json 改为 _step2_filter_rule_based.json
        stem = input_path.name[: -len(CONTENT_LIST_V2_SUFFIX)]
        out_path = input_path.parent / (stem + STEP2_OUTPUT_SUFFIX)
        try:
            process_one_file(
                input_path,
                out_path,
                flatten_lists=args.flatten_lists,
                print_items=args.print_items,
            )
            print(f"[{i}/{len(files)}] 已处理并写入：{out_path}")
        except Exception as e:
            print(f"[{i}/{len(files)}] 处理失败 {input_path}：{e}", file=sys.stderr)
            sys.exit(1)

    print("全部处理完成。")


if __name__ == "__main__":
    main()