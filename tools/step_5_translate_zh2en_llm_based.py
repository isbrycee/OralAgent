import os
import json
from openai import OpenAI

# ========= 初始化 OpenAI 客户端 =========
# 请先在环境变量中设置 OPENAI_API_KEY
#   export OPENAI_API_KEY="你的 API Key"
client = OpenAI(
    api_key="sk-",
    base_url="https://www.dmxapi.cn/v1/"
)

# ========= 详细的翻译系统提示词（SYSTEM PROMPT） =========
SYSTEM_PROMPT = """
You are a professional bilingual translator specializing in
Chinese-to-English translation of medical and dental academic texts,
especially in the fields of stomatology, oral medicine, and related
biomedical sciences.

Your responsibilities:
- Translate Chinese text into accurate, professional, natural-sounding English.
- Preserve the full meaning of the original text without adding, removing, or altering information.
- Use precise and field-specific terminology for dentistry, oral medicine, and biomedical sciences.

Output requirements:
- You will receive one piece of Chinese text at a time.
- Your output must consist solely of the English translation of the text.
- Do not include any explanations, comments, or additional content.
- Do not repeat the original Chinese text.
- Do not add any information unrelated to the translation.
"""

# ========= 用户提示（可选，简单说明任务） =========
USER_INSTRUCTION = """
Translate the following Chinese dental-related text into professional and accurate English.
Provide only the translation, without explanations or additional comments.
"""

def translate_text_with_gpt(text: str, model_name: str = "gpt-4.1-mini") -> str:
    """
    调用 GPT，将一段中文文本翻译成英文。
    一次只翻译一个文本，不做 batch。
    """
    if not text.strip():
        return ""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": USER_INSTRUCTION.strip() + "\n\n" + text.strip(),
            },
        ],
    )

    translation = response.choices[0].message.content.strip()
    return translation


def process_json_file(
    input_path: str,
    output_path: str | None = None,
    model_name: str = "gpt-4.1-mini",
):
    """
    处理单个 JSON 文件：
    - 读取为 list[dict]，每个 dict 至少包含 text 字段
    - 对其中的 text 字段逐条调用 GPT 翻译
    - 将翻译写入 text_en 字段
    - 保存到 output_path（默认 input_path 后加 _en）
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_en" + ext

    # 读取 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"文件 {input_path} 的 JSON 顶层结构不是 list")

    # 逐条翻译
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"文件 {input_path} 中第 {idx} 个元素不是 dict：{item}")

        src_text = item.get("text", "")
        if not isinstance(src_text, str):
            raise ValueError(
                f"文件 {input_path} 中第 {idx} 个元素的 text 不是字符串：{src_text}"
            )

        if src_text.strip():
            print(f"翻译 {os.path.basename(input_path)} 第 {idx} 段文本...")
            en_text = translate_text_with_gpt(src_text, model_name=model_name)
        else:
            en_text = ""

        item["text_en"] = en_text

    # 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"处理完成：{input_path} -> {output_path}")


def process_folder(folder_path: str, model_name: str = "gpt-4.1-mini"):
    """
    扫描文件夹中所有以 '_chunk_merged.json' 结尾的文件并逐个处理。
    """
    for filename in os.listdir(folder_path):
        if not filename.endswith("_chunk_merged.json"):
            continue

        input_path = os.path.join(folder_path, filename)

        # 如需覆盖原文件，可改为 output_path=input_path
        process_json_file(
            input_path=input_path,
            output_path = input_path.replace(".json", "_translated_zh2en.json"),  # 添加后缀 "_translated"
            model_name=model_name,
        )


if __name__ == "__main__":
    # 使用示例：替换为你的 json 文件夹路径
    folder = "/path/to/your/json_folder"
    process_folder(folder_path=folder, model_name="gpt-5-nano")

    