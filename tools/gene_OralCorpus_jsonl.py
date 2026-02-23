"""
将 root_dir/类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned_w_meta.json 中的 items
按条提取为 JSONL，每个 JSON 生成中文版、英文版两个 JSONL 文件，分别保存到指定根目录下的两个文件夹。
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

HYBRID_AUTO_DIR = "hybrid_auto"
STEP3_W_META_SUFFIX = "_step4_sim_chunk.json"

# 跳过的书名（与目录名一致，即 类别/语言目录/书名 中的 书名）
SKIP_BOOK_NAMES = frozenset({
    "口腔修复工艺学（北医第2版2020）__周永胜",
    "(ITItreatmentguidevol2)牙种植学的负荷方案：牙列缺损的负荷方案__D.Wismeijer",
    "国际口腔种植学会ITI第三卷口腔种植临床指南-拔牙位点种植：各种治疗方案__S.Chen",
})


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_filename(s: str) -> str:
    """将字符串转为可作文件名的形式，去掉或替换非法字符。"""
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = s.strip().strip(".")
    return s[:200] if len(s) > 200 else (s or "unnamed")


def discover_step3_w_meta_files(root_dir: Path) -> List[Path]:
    """
    在根目录下查找所有 hybrid_auto/*_step3_llm_cleaned_w_meta.json 文件。
    目录结构：root_dir / 类别 / 语言目录 / 书名 / hybrid_auto / *_step3_llm_cleaned_w_meta.json
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
                    if f.is_file() and f.name.endswith(STEP3_W_META_SUFFIX):
                        found.append(f)
    return sorted(found)


def json_to_jsonl_records(
    data: Dict[str, Any],
    lang: str,
) -> List[Dict[str, Any]]:
    """
    从 JSON 的 items 和 meta 生成 JSONL 行记录列表。
    lang: "ch" 使用 subject_ch, title_ch, text_ch；"en" 使用 subject_en, title_en, text_en。
    原版语言（Init_lang）均来自 meta["Init_lang"]。
    """
    items = data.get("items") or []
    meta = data.get("meta") or {}
    subject_id = str(meta.get("subject_id", ""))
    subject_ch = meta.get("subject_ch", "")
    subject_en = meta.get("subject_en", "")
    title_ch = meta.get("title_ch", "")
    title_en = meta.get("title_en", "")
    init_lang = meta.get("Init_lang", "")

    if lang == "ch":
        subject = subject_ch
        title = title_ch
        text_key = "text_ch"
    else:
        subject = subject_en
        title = title_en
        text_key = "text_en"

    # 中文版与英文版使用不同的 key 名
    keys_ch = ("ID", "学科", "学科_ID", "书名", "页码", "内容", "原版语言")
    keys_en = ("ID", "Subject", "Subject_ID", "Title", "Page_number", "Content", "Init_lang")

    records: List[Dict[str, Any]] = []
    record_id = 1
    for item in items:
        if not isinstance(item, dict):
            continue
        page_number = item.get("page_number")
        text = item.get(text_key, "")
        values = (record_id, subject, subject_id, title, page_number, text, init_lang)
        record_id += 1
        if lang == "ch":
            records.append(dict(zip(keys_ch, values)))
        else:
            records.append(dict(zip(keys_en, values)))
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(
    root_dir: Path,
    out_root: Path,
    ch_dir_name: str = "CH",
    en_dir_name: str = "EN",
) -> None:
    """
    root_dir: 扫描的根目录（类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned_w_meta.json）
    out_root: 保存 JSONL 的根目录；其下 ch_dir_name、en_dir_name 分别为中文版、英文版子目录。
    """
    files = discover_step3_w_meta_files(root_dir)
    if not files:
        print(
            f"在 {root_dir} 下未找到任何 类别/语言目录/书名/{HYBRID_AUTO_DIR}/*{STEP3_W_META_SUFFIX} 文件。",
            file=sys.stderr,
        )
        sys.exit(1)

    out_ch = out_root / ch_dir_name
    out_en = out_root / en_dir_name

    for json_path in tqdm(files, desc="生成 JSONL", unit="个"):
        # 书名即该 json 所在路径的上一级目录名：.../书名/hybrid_auto/xxx.json
        book_name = json_path.parent.parent.name
        if book_name in SKIP_BOOK_NAMES:
            continue
        try:
            data = load_json(str(json_path))
        except Exception as e:
            print(f"[WARN] 读取失败 {json_path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            continue
        items = data.get("items") or []
        if len(items) == 0:
            continue
        meta = data.get("meta") or {}
        subject_id = str(meta.get("subject_id", ""))
        title_ch = meta.get("title_ch", "")
        title_en = meta.get("title_en", "")

        records_ch = json_to_jsonl_records(data, "ch")
        records_en = json_to_jsonl_records(data, "en")

        base_ch = f"{subject_id}_{safe_filename(title_ch)}.jsonl"
        base_en = f"{subject_id}_{safe_filename(title_en)}.jsonl"
        write_jsonl(out_ch / base_ch, records_ch)
        write_jsonl(out_en / base_en, records_en)


def main():
    parser = argparse.ArgumentParser(
        description="将 step3_llm_cleaned_w_meta.json 的 items 提取为中文/英文两份 JSONL，保存到指定根目录下的两个子目录。"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="扫描根目录。将扫描 root_dir/类别/语言目录/书名/hybrid_auto/*_step3_llm_cleaned_w_meta.json。",
    )
    parser.add_argument(
        "out_root",
        type=Path,
        help="保存 JSONL 的根目录；其下将创建中文版、英文版两个子目录。",
    )
    parser.add_argument(
        "--ch-dir",
        default="CH",
        help="中文版 JSONL 子目录名（默认: CH）。",
    )
    parser.add_argument(
        "--en-dir",
        default="EN",
        help="英文版 JSONL 子目录名（默认: EN）。",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        print(f"根目录不存在或不是目录：{root}", file=sys.stderr)
        sys.exit(1)

    run(
        root,
        args.out_root.resolve(),
        ch_dir_name=args.ch_dir,
        en_dir_name=args.en_dir,
    )


if __name__ == "__main__":
    main()
