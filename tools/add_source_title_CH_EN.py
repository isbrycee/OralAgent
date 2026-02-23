"""
为 root_dir/类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned.json 添加 meta 字段，
并保存为同目录下的 *_step3_llm_cleaned_w_meta.json。

Excel 三列：当前书名、润色后中文书名、润色后英文书名。
meta 字段：title_ch, title_en, subject_ch, subject_en, Init_lang。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

HYBRID_AUTO_DIR = "hybrid_auto"
STEP3_LLM_CLEANED_SUFFIX = "_step3_llm_cleaned.json"
STEP3_W_META_SUFFIX = "_step3_llm_cleaned_w_meta.json"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_title_excel(excel_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    读取 Excel：第一行为表头，前三列依次为当前书名、润色后中文书名、润色后英文书名。
    返回 当前书名 -> (title_ch, title_en)。
    """
    df = pd.read_excel(excel_path, header=0)
    if df.shape[1] < 3:
        raise ValueError(f"Excel 至少需要 3 列，当前只有 {df.shape[1]} 列")
    mapping: Dict[str, Tuple[str, str]] = {}
    for _, row in df.iterrows():
        key = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        ch = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        en = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        if key:
            mapping[key] = (ch, en)
    return mapping


def parse_category_subject(category_name: str) -> Tuple[str, str]:
    """
    从类别名解析 subject_ch 和 subject_en。
    例如 '11_口腔生物学_Oral Biology' -> subject_ch='口腔生物学', subject_en='Oral Biology'
    - subject_ch: 第一个下划线与第二个下划线之间的内容
    - subject_en: 最后一个下划线之后的内容
    """
    parts = category_name.split("_")
    if len(parts) < 2:
        return ("", "", "")
    subject_id = parts[0] if len(parts) > 1 else ""
    subject_ch = parts[1] if len(parts) > 1 else ""
    subject_en = parts[-1] if len(parts) > 1 else ""
    return (subject_id, subject_ch, subject_en)


def parse_init_lang(lang_dir_name: str) -> str:
    """
    从语言目录名得到初始语言。
    包含 'CH' -> 'Chinese'，包含 'EN' -> 'English'，否则返回空字符串。
    """
    u = lang_dir_name.upper()
    if "CH" in u:
        return "Chinese"
    if "EN" in u:
        return "English"
    return ""


def discover_step3_llm_cleaned_files(root_dir: Path) -> List[Path]:
    """
    在根目录下查找所有 hybrid_auto/*_step3_llm_cleaned.json 文件。
    目录结构：root_dir / 类别 / 语言目录 / 书名 / hybrid_auto / *_step3_llm_cleaned.json
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


def path_to_meta(
    json_path: Path,
    root_dir: Path,
    title_map: Dict[str, Tuple[str, str]],
) -> Dict[str, str]:
    """
    根据文件路径和 Excel 映射生成 meta 字段。
    路径形式：root_dir / 类别 / 语言目录 / 书名 / hybrid_auto / xxx_step3_llm_cleaned.json
    """
    root = Path(root_dir).resolve()
    try:
        rel = json_path.relative_to(root)
    except ValueError:
        return {}
    parts = rel.parts
    if len(parts) < 4:
        return {}
    category_name = parts[0]
    lang_dir_name = parts[1]
    book_name = parts[2]

    subject_id, subject_ch, subject_en = parse_category_subject(category_name)
    init_lang = parse_init_lang(lang_dir_name)
    title_ch, title_en = title_map.get(book_name, ("", ""))

    return {
        "title_ch": title_ch,
        "title_en": title_en,
        "subject_id": subject_id,
        "subject_ch": subject_ch,
        "subject_en": subject_en,
        "Init_lang": init_lang,
    }


def add_meta_and_save(
    root_dir: Path,
    excel_path: Path,
    dry_run: bool = False,
) -> None:
    title_map = load_title_excel(excel_path)
    files = discover_step3_llm_cleaned_files(root_dir)
    if not files:
        print(
            f"在 {root_dir} 下未找到任何 类别/语言目录/书名/{HYBRID_AUTO_DIR}/*{STEP3_LLM_CLEANED_SUFFIX} 文件。",
            file=sys.stderr,
        )
        sys.exit(1)

    for json_path in tqdm(files, desc="添加 meta 并保存", unit="个"):
        meta = path_to_meta(json_path, root_dir, title_map)
        try:
            data = load_json(str(json_path))
        except Exception as e:
            print(f"[WARN] 读取失败 {json_path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            continue
        # 保留 step3 原始数据，仅在 meta 中新增或更新指定字段
        if "meta" not in data:
            data["meta"] = {}
        data["meta"].update(meta)

        out_name = json_path.name.replace(STEP3_LLM_CLEANED_SUFFIX, STEP3_W_META_SUFFIX)
        out_path = json_path.parent / out_name
        if not dry_run:
            save_json(str(out_path), data)
        else:
            print(f"[dry-run] 将写入: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="为 step3 LLM 清洗后的 JSON 添加 title_ch/title_en/subject_ch/subject_en/Init_lang 等 meta，并保存为 *_step3_llm_cleaned_w_meta.json。"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="根目录。将扫描 root_dir/类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned.json。",
    )
    parser.add_argument(
        "excel",
        type=Path,
        help="Excel 路径，三列：当前书名、润色后中文书名、润色后英文书名。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要写入的文件，不实际写入。",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        print(f"根目录不存在或不是目录：{root}", file=sys.stderr)
        sys.exit(1)
    if not args.excel.is_file():
        print(f"Excel 文件不存在：{args.excel}", file=sys.stderr)
        sys.exit(1)

    add_meta_and_save(root, args.excel.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
