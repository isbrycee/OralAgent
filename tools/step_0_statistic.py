#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="递归查找以 '_content_list.json' 结尾的文件并统计总数"
    )
    parser.add_argument("root_dir", help="数据集根目录路径")
    parser.add_argument(
        "--absolute", "-a", action="store_true",
        help="以绝对路径输出（默认输出相对 root_dir 的路径）"
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root_dir))

    if not os.path.exists(root_dir):
        print(f"路径不存在: {root_dir}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(root_dir):
        print(f"不是目录: {root_dir}", file=sys.stderr)
        sys.exit(2)

    total = 0

    def onerror(err):
        print(f"无法访问目录: {err}", file=sys.stderr)

    for cur_root, dirs, files in os.walk(root_dir, topdown=True, onerror=onerror, followlinks=False):
        for name in files:
            if name.endswith("_content_list.json"):
                total += 1
                full_path = os.path.join(cur_root, name)
                if args.absolute:
                    print(full_path)
                else:
                    rel_path = os.path.relpath(full_path, root_dir)
                    print(rel_path)

    print(f"总数量: {total}")

if __name__ == "__main__":
    main()
