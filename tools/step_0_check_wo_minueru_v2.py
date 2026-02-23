#!/usr/bin/env python3
"""
检查根目录下所有 hybrid_auto 目录是否包含以 _content_list_v2.json 结尾的文件。
打印出没有该 JSON 文件的文件夹路径。
"""

import os
import sys
import argparse


def has_content_list_v2(directory: str) -> bool:
    """检查目录下是否存在以 _content_list_v2.json 结尾的文件。"""
    if not os.path.isdir(directory):
        return False
    for name in os.listdir(directory):
        if name.endswith("_content_list_v2.json"):
            return True
    return False


def find_hybrid_auto_dirs(root: str) -> list[str]:
    """在根目录下递归查找所有名为 hybrid_auto 的目录。"""
    result = []
    for dirpath, dirnames, _ in os.walk(root):
        if "hybrid_auto" in dirnames:
            hybrid_path = os.path.join(dirpath, "hybrid_auto")
            result.append(hybrid_path)
            # 不再进入 hybrid_auto 子目录继续遍历
            dirnames.remove("hybrid_auto")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="检查根目录下所有 hybrid_auto 目录是否包含 _content_list_v2.json 文件"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="根目录路径（默认当前目录）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="同时打印有该文件的目录数量",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"错误：路径不是目录或不存在: {root}", file=sys.stderr)
        sys.exit(1)

    hybrid_dirs = find_hybrid_auto_dirs(root)
    missing = []
    has_file = []

    for d in hybrid_dirs:
        if has_content_list_v2(d):
            has_file.append(d)
        else:
            missing.append(d)

    # 打印没有 _content_list_v2.json 的目录
    if missing:
        print(f"以下 {len(missing)} 个 hybrid_auto 目录下没有以 _content_list_v2.json 结尾的文件：\n")
        for path in sorted(missing):
            print(path)
    else:
        print("所有 hybrid_auto 目录下均存在 _content_list_v2.json 文件。")

    if args.verbose:
        print(f"\n统计：共 {len(hybrid_dirs)} 个 hybrid_auto 目录，"
              f"{len(has_file)} 个有该文件，{len(missing)} 个缺失。")


if __name__ == "__main__":
    main()

