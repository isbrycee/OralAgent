#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

try:
    from openpyxl import Workbook
except ImportError:
    print("缺少依赖 openpyxl，请先执行: pip install openpyxl", file=sys.stderr)
    sys.exit(1)


def collect_pdf_titles_fixed_depth(root_dir: str):
    """
    从 root_dir/类别目录/子目录/*.pdf 收集 PDF 文件名（去掉扩展名）。
    只匹配“根目录下两级子目录中的 PDF”，不做更深层递归。
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"根目录不存在或不是目录: {root}")

    titles = []
    # 第一层：类别目录
    for category in root.iterdir():
        if not category.is_dir():
            continue
        # 第二层：子目录
        for sub in category.iterdir():
            if not sub.is_dir():
                continue
            # 只遍历该子目录内直接包含的文件
            for f in sub.iterdir():
                if f.is_file() and f.suffix.lower() == ".pdf":
                    titles.append(f.stem)
    return titles


def save_to_excel(titles, output_path: str):
    """
    将标题列表写入 Excel 第一列，首行标题为“书名”。
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "books"
    ws.append(["书名"])
    for t in titles:
        ws.append([t])

    wb.save(out)


def main():
    parser = argparse.ArgumentParser(
        description="从 /data/OralGPT/OralGPT-text-corpus/类别/子目录/*.pdf 提取文件名（去掉 .pdf）到 Excel"
    )
    parser.add_argument(
        "root_dir",
        help="根目录，例如：/data/OralGPT/OralGPT-text-corpus/",
    )
    parser.add_argument(
        "-o", "--output", default="books.xlsx", help="输出 Excel 文件路径（默认：books.xlsx）"
    )
    args = parser.parse_args()

    try:
        titles = collect_pdf_titles_fixed_depth(args.root_dir)
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    save_to_excel(titles, args.output)
    print(f"共找到 {len(titles)} 个条目，已写入：{args.output}")


if __name__ == "__main__":
    main()