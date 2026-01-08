import os
import json

def clean_json_file(input_path: str, output_path: str):
    # 计数器
    cnt_text_level_1 = 0
    cnt_type_filtered = 0
    cnt_empty_text = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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

    # 打印三种类型的过滤数量
    print("过滤统计：")
    print(f"  text_level == 1: {cnt_text_level_1}")
    print(f"  type in ('image', 'table'): {cnt_type_filtered}")
    print(f"  空文本（或无 text 字段）: {cnt_empty_text}")
    
    # 将清洗后的数据写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)


def clean_json_folder(input_dir: str, output_dir: str):
    """遍历文件夹，对其中所有 .json 文件进行清洗"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".json"):
            continue

        input_path = os.path.join(input_dir, filename)

        name, ext = os.path.splitext(filename)
        filename = f"{name}_filter_rule_based{ext}"
        output_path = os.path.join(output_dir, filename)

        try:
            clean_json_file(input_path, output_path)
            print(f"已处理：{filename}")
            print("\n")
        except Exception as e:
            print(f"处理 {filename} 时出错：{e}")


if __name__ == "__main__":
    # 使用示例：修改为你自己的路径
    input_folder = r"/home/jinghao/projects/OralGPT-Agent/Corpus/ocr_demo"   # 原始 JSON 文件夹
    output_folder = r"/home/jinghao/projects/OralGPT-Agent/Corpus/ocr_demo" # 清洗后 JSON 文件夹

    clean_json_folder(input_folder, output_folder)
