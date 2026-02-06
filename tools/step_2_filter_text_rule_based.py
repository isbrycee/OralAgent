# import os
# import json

# def clean_json_file(input_path: str, output_path: str):
#     # 计数器
#     cnt_text_level_1 = 0
#     cnt_type_filtered = 0
#     cnt_empty_text = 0

#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     if not isinstance(data, list):
#         print(f"跳过（不是列表）：{input_path}")
#         return

#     cleaned = []
#     for item in data:
#         if not isinstance(item, dict):
#             continue

#         # 1. text_level == 1
#         if item.get("text_level") == 1:
#             cnt_text_level_1 += 1
#             continue

#         # 2. type == image / table
#         if item.get("type") in ("image", "table"):
#             cnt_type_filtered += 1
#             continue

#         # 3. 空 text
#         text = item.get("text")
#         if not isinstance(text, str) or text.strip() == "":
#             cnt_empty_text += 1
#             continue

#         item["text"] = text.strip()
#         cleaned.append(item)

#     # 打印三种类型的过滤数量
#     print("过滤统计：")
#     print(f"  text_level == 1: {cnt_text_level_1}")
#     print(f"  type in ('image', 'table'): {cnt_type_filtered}")
#     print(f"  空文本（或无 text 字段）: {cnt_empty_text}")
    
#     # 将清洗后的数据写入新文件
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(cleaned, f, ensure_ascii=False, indent=2)


# def clean_json_folder(input_dir: str, output_dir: str):
#     """遍历文件夹，对其中所有 .json 文件进行清洗"""
#     os.makedirs(output_dir, exist_ok=True)

#     for filename in os.listdir(input_dir):
#         if not filename.lower().endswith(".json"):
#             continue

#         input_path = os.path.join(input_dir, filename)

#         name, ext = os.path.splitext(filename)
#         filename = f"{name}_filter_rule_based{ext}"
#         output_path = os.path.join(output_dir, filename)
        
#         try:
#             clean_json_file(input_path, output_path)
#             print(f"已处理：{filename}")
#             print("\n")
#         except Exception as e:
#             print(f"处理 {filename} 时出错：{e}")


# if __name__ == "__main__":
#     # 使用示例：修改为你自己的路径
#     input_folder = r"/home/jinghao/projects/OralGPT-Agent/Corpus/ocr_demo"   # 原始 JSON 文件夹
#     output_folder = r"/home/jinghao/projects/OralGPT-Agent/Corpus/ocr_demo" # 清洗后 JSON 文件夹

#     clean_json_folder(input_folder, output_folder)

import os
import json

def clean_json_file(input_path: str, output_path: str):
    # 计数器
    cnt_text_level_1 = 0
    cnt_type_filtered = 0
    cnt_empty_text = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取失败：{input_path} -> {e}")
        return

    if not isinstance(data, list):
        print(f"跳过（不是列表）：{input_path}")
        return

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue

        # 1. text_level == 1
        if item.get("text_level") == 1:
            cnt_text_level_1 += 1
            continue

        # 2. type == image / table
        if item.get("type") in ("image", "table"):
            cnt_type_filtered += 1
            continue

        # 3. 空 text
        text = item.get("text")
        if not isinstance(text, str) or text.strip() == "":
            cnt_empty_text += 1
            continue

        item["text"] = text.strip()
        cleaned.append(item)

    # 打印过滤统计
    print("过滤统计：")
    print(f"  text_level == 1: {cnt_text_level_1}")
    print(f"  type in ('image', 'table'): {cnt_type_filtered}")
    print(f"  空文本（或无 text 字段）: {cnt_empty_text}")
    
    # 写出
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"写入失败：{output_path} -> {e}")


def looks_like_content_list(filename: str) -> bool:
    """
    判断文件名是否为目标 content_list 文件：
    - 名字中包含 'content_list.json'（大小写不敏感）
    - 且不是已生成的 *_filter_rule_based.json
    """
    lower = filename.lower()
    if "_filter_rule_based.json" in lower:
        return False
    return "content_list.json" in lower


def clean_content_lists_in_middle(root_dir: str, followlinks: bool = True):
    """
    递归遍历 root_dir 的所有目录层级。
    在每个目录层级：
      - 如果该目录里存在 content_list.json（包含式匹配），则对其进行清洗并输出到同目录（加后缀）。
      - 找到后不再进入该目录的子目录（避免进一步深入该“中间层”的子目录）。
    """
    if not os.path.isdir(root_dir):
        print(f"提供的路径不是目录：{root_dir}")
        return

    found = 0

    # topdown=True 以便在找到目标后通过清空 dirnames 来“剪枝”
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True, followlinks=followlinks):
        # 当前目录下找匹配文件
        matches = [fn for fn in filenames if looks_like_content_list(fn)]
        if matches:
            for filename in matches:
                input_path = os.path.join(dirpath, filename)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_step2_filter_rule_based{ext}"
                output_path = os.path.join(dirpath, output_filename)
                try:
                    print(f"开始处理：{input_path}")
                    clean_json_file(input_path, output_path)
                    print(f"已处理并保存至：{output_path}\n")
                    found += 1
                except Exception as e:
                    print(f"处理 {input_path} 时出错：{e}")

            # 关键点：该目录已经是我们要的“中间层”，不再深入其子目录
            dirnames[:] = []
            continue

        # 如果当前目录没有匹配文件，则继续常规递归（不剪枝）

    if found == 0:
        print("未在目录中找到任何 content_list.json（名字包含式匹配）。")


if __name__ == "__main__":
    # 修改为你的根目录路径
    root_folder = r"/data/OralGPT/OralGPT-text-corpus-processed"
    clean_content_lists_in_middle(root_folder)
