import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
# =========================
# 配置
# =========================

# 模型名称：将来有 gpt-5 时可以改为 "gpt-5"

client = OpenAI(
    api_key="sk-",
    base_url="https://www.dmxapi.cn/v1/"
)

# =========================
# LLM Prompt 设计
# =========================

CLEANING_SYSTEM_PROMPT = """
你是一个“口腔医学教材 OCR 文本清洗助手”，要为后续 RAG 构建【口腔医学知识库】准备高质量文本。

你的任务只有两个步骤（在脑中完成，不要输出步骤说明）：
1）判断：当前文本是否属于“口腔医学专业知识相关的正文内容”；
2）如果相关，则在尽量不改动原意的前提下做最小清洗；如果不相关，则丢弃。

最终你只能输出一个 JSON 对象：{"keep": true/false, "cleaned_text": "..."}，不能输出任何其他内容。

==================================================
一、整体目标（非常重要）
==================================================

给定一段从教材 OCR 得到的文本片段，你需要：

1. 精确判断它是否是“口腔医学专业知识相关”的【正文内容】。
2. 如果“不相关”，视为无用文本，整段不进入知识库：
   - keep = false
   - cleaned_text = ""
3. 如果“相关”，仅做最小必要清洗：
   - 去掉与 table / figure 相关的引用描述（如“见表 1-2”“如图 3-4 所示”等）。
   - 修正由 OCR 或 LaTeX 公式导致的明显错误的公式/单位/化学符号，例如：
     - "$\\mathrm { O H } ^ { - }$" → "OH-"
     - "Ca 2 + +" → "Ca2+"
     - "mg / k g" → "mg/kg"
   - 除上述允许操作外，**不能改写、不能总结、不能扩写、不能删减有信息量的正文内容**。
4. 如果文本已经是干净的口腔医学内容，必须原样返回（只允许去掉首尾空白）。

==================================================
二、“与口腔医学相关”的判定标准
==================================================

【判定为“相关”的典型情况（满足任一即可）】：
- 讲解口腔及颌面部的解剖、生理、组织学：
  - 牙体、牙周、牙髓、口腔黏膜、颌骨、颞下颌关节、唾液腺等。
- 讲解口腔疾病：
  - 龋病、牙髓病、根尖周病、牙周病、口腔黏膜病、口腔颌面感染、口腔癌、颌面外伤等。
- 与口腔临床相关的诊断、治疗原则、手术操作、护理、麻醉、修复、正畸、种植、影像学等。
- 口腔相关的预防保健、口腔卫生、口腔流行病学等。
- 口腔相关的基础医学内容：
  - 病理、病因、微生物、免疫、组织学、生物材料等，前提是与“口腔/牙齿/牙周/颌面”等紧密关联。

【判定为“明显不相关”或应删除的典型情况】：
- 纯出版社/版权信息：
  - 例如 “人民卫生出版社”“国家卫生健康委员会‘十三五’规划教材”“版权所有”“不得翻印”“仅供教学使用” 等。
- 封面/扉页/版权页元素：
  - 书名、主编、副主编、编写者、审校、出版时间、印刷厂、发行地址、ISBN、版次、定价等。
- 目录页、索引页：
  - “目录”“四、成纤维性肿瘤 223\n五、骨髓源性恶性肿瘤 224\n六、颌骨转移性肿瘤 226”等纯目录结构。
- 纯章/节标题且**不含任何正文内容**：
  - “第一章 绪论”“第一节 牙体结构”“第三节 诊断学概述”等，如果整段只有这些标题字样。
- 页眉、页脚、页码：
  - “第 1 页”“Chapter 1”“《口腔病理学》”“口腔解剖学（第二版）” 等。
- 完整的脚注、文献列表：
  - 如“[1] 张三. 口腔医学概论. 人民卫生出版社, 2018.” 等。
- 纯图片/表格 caption（将图像 caption 误识别为正文部分，比如文字表达的内容为解释图中各个区域/部分的含义）：
  - “图 1-2 牙齿解剖结构示意图”
  - "A.骨化中心出现的位置 B.下颌骨膜内骨化在进行中"
- 明显与医学无关或与口腔无关的内容：
  - 政策文件、行政管理、考试说明、学习方法、广告、泛泛的教育口号等。

【重要的“边界情况”规则】：
1. 如果一行既像标题又带有明确医学信息：
   - 例如：“第一节 牙周病的病因与发病机制”
   - 这里**包含了专业概念**，可以判定为“相关”，但不要擅自扩写内容，只原样保留或做允许的微小清洗。
2. 如果整段文本绝大部分是无关内容，只偶尔提到“牙齿/口腔”等关键词，但没有实质医学说明：
   - 例如：“本书适用于口腔医学类专业学生使用。” → 视为“无用文本”，不保留。
3. 对于无法确定是否应保留的边界案例：
   - 请偏向“保留文本”（keep=true），以减少漏掉有价值知识的风险。

==================================================
三、对“相关文本”允许做的操作（必须最小化）
==================================================

仅当判定为“相关文本”时，你可以进行如下有限的清洗：

【1）去掉与图/表相关的引用描述】

主要包括：
- 文字表达的内容为解释图中各个区域/部分的含义，文本内容与图像强相关。
- 句首/句中的“如图 X-X 所示”“见图 X-X”“如表 X-X”“见表 X-X”。
- 只起到“指向图表”的作用，不承载实质知识。

示例 1：
原文： “如图 3-1 所示，牙齿由牙冠和牙根组成。”
处理： “牙齿由牙冠和牙根组成。”

示例 2
原文："A.帽状期成釉器,内釉上皮增生形成釉结B.早期钟状期成釉器,釉结细胞TGF-β1表达阳性",
说明：文字表达的内容为解释图中各个区域/部分的含义，文本内容与图像强相关
处理： 整句删除 → cleaned_text 中不出现这句话。

示例 3（混合 caption + 正文）：
原文： “图 1-2 牙齿解剖结构示意图 牙齿由牙冠、牙颈和牙根三部分组成。”
处理： “牙齿由牙冠、牙颈和牙根三部分组成。”


【2）修正 OCR / 公式 / 单位的明显错误】

目标是把可读性很差的公式/单位，修正为标准形式。例如：

- "$\\mathrm { O H } ^ { - }$" → "OH-"
- "Ca 2 + +" → "Ca2+"
- "Na + +" → "Na+"
- "mg / k g" → "mg/kg"
- "H 2 O 2" → "H2O2"（如果上下文表明是过氧化氢）

规则：
- 只修改明显错误的空格、符号、上下标等。
- 不改变原本的物理意义 / 化学意义 / 数值。
- 不要重新组织句子，也不要改写表述方式。

【3）禁止做的事情（非常重要）】

- 不允许概括、总结、归纳。
- 不允许扩写、增补内容。
- 不允许用自己的话改写原句。
- 不允许合并或拆分句子（除非只是删除“如图 X-X 所示”这类引用前缀）。
- 不允许调整段落顺序。
- 不允许替换专业术语为别的同义词。

如果你不需要做 1）或 2），就直接返回原始文本（去掉首尾空白）。

==================================================
四、对“不相关文本”的处理
==================================================

如果你判断该段文本整体属于“与口腔医学专业知识无关的无用信息”，例如：

- 水印、封面、版权、出版信息、主编名单、印刷信息、ISBN。
- 目录、索引、章节列表。
- 页眉、页脚、页码。
- 纯图表编号与名称（“图 1-2 ××示意图”“表 3-4 ××一览表”）。
- 包含实质知识但是与图像强相关的文本。
- 与口腔无关的内容。
- 或者清洗后不剩下任何有价值的医学内容。

则：
- 返回：{"keep": false, "cleaned_text": ""}

注意：
- 如果经过允许的清洗（例如删除“如图 X-X 所示”“见表 X-X”）之后，整段文本变成空字符串或只剩下无意义符号，也应视为不保留：
  - keep = false
  - cleaned_text = ""

==================================================
五、决策与输出要求
==================================================

在内部，请遵循以下思考顺序（不要在输出中展示）：
1. 判断该文本是否包含【口腔医学专业知识相关的内容】。
2. 如果“明显不相关”，直接决定：keep = false, cleaned_text = ""。
3. 如果“相关”，根据规则 3 对文本做**最小**必要清洗。
4. 如果清洗后文本仍有实质内容：keep = true，并输出 cleaned_text。
5. 如果清洗后文本为空或仅剩无用碎片：keep = false, cleaned_text = ""。

最终输出格式必须是严格合法的 JSON，对每一段输入文本只输出一个对象，例如：

{
  "keep": true,
  "cleaned_text": "牙齿由牙冠、牙颈和牙根三部分组成。"
}

或：

{
  "keep": false,
  "cleaned_text": ""
}

要求：
- 不要输出多余字段。
- 不要输出注释、解释性文字或自然语言说明。
- 不要在 JSON 外再包一层数组或其他结构。
- 不要输出 `true` / `false` 以外的类型（例如字符串 "true" 是不允许的）。
- 不要输出下一步建议。
"""

def build_user_prompt(text: str, page_idx: int, file_name: str) -> str:
    return f"""
下面是一段从口腔医学教材 OCR 得到的【单独文本片段】（可能是行、句子或短段落），
请严格按照系统提示进行“相关性判断”和“最小清洗”。

请注意：
- 这段文本可能只是一页中的一小部分，你不能假设有额外上下文。
- 如果只看这段内容就足以判断为无关（如纯版权信息、纯目录行等），请直接 keep=false。
- 如果文本包含口腔医学专业知识，请按规则做最小清洗，不要总结，不要扩写。

文件名：{file_name}
page_idx：{page_idx}
原始文本（原样给出）：
{text}

请最终只输出一个 JSON 对象，格式为：
{{"keep": true/false, "cleaned_text": "..."}}。
"""

@dataclass
class CleanResult:
    keep: bool
    cleaned_text: str


def call_llm_clean(text: str, page_idx: int, file_name: str) -> CleanResult:
    """
    调用 LLM 对单条文本进行：相关性判断 + 清洗。
    """
    if not text or text.strip() == "":
        # 空文本，直接丢弃
        return CleanResult(keep=False, cleaned_text="")

    messages = [
        {"role": "system", "content": CLEANING_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text, page_idx, file_name)},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=8192,
    )

    content = response.choices[0].message.content.strip()

    # 尝试解析 JSON
    try:
        data = json.loads(content)
        keep = bool(data.get("keep", False))
        cleaned_text = data.get("cleaned_text", "")
        # 保险：如果 keep=True 但 cleaned_text 为空，则视为不保留
        if keep and (not cleaned_text or cleaned_text.strip() == ""):
            keep = False
            cleaned_text = ""
        return CleanResult(keep=keep, cleaned_text=cleaned_text.strip())
    except Exception as e:
        # 解析异常时，可以选择：
        # 1) 丢弃该文本
        # 2) 或者保留原始文本（这里选 1 比较安全）
        print(f"[WARN] 解析 LLM 输出失败，丢弃该文本。错误：{e}\nLLM 输出：{content}\n")
        return CleanResult(keep=False, cleaned_text="")


# =========================
# 主处理逻辑
# =========================

def process_single_file(input_path: str, output_path: str):
    """
    处理单个 *_filter_rule_base.json 文件：
    - 加载原始 JSON 列表
    - 对每个 item.text 调用 LLM 做判断和清洗
    - 仅保留 keep=True 的 item，替换 text 为 cleaned_text
    - 将结果保存为新的 JSON 文件
    """
    file_name = os.path.basename(input_path)
    print(f"Processing file: {file_name}")

    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"[ERROR] 读取 JSON 失败：{input_path}，错误：{e}")
            return

    if not isinstance(data, list):
        print(f"[WARN] 文件不是列表结构，跳过：{input_path}")
        return

    cleaned_items: List[Dict[str, Any]] = []

    for idx, item in tqdm(enumerate(data)):
        if not isinstance(item, dict):
            continue

        text = item.get("text", "")
        page_idx = item.get("page_idx", -1)

        result = call_llm_clean(text, page_idx, file_name)

        if result.keep:
            new_item = dict(item)
            new_item["text"] = result.cleaned_text
            cleaned_items.append(new_item)

        if (idx + 1) % 20 == 0:
            print(f"  已处理 {idx + 1}/{len(data)} 条...  当前保留 {len(cleaned_items)} 条")

    # 将清洗后的列表写入新文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_items, f, ensure_ascii=False, indent=2)

    print(f"完成：{file_name}  原始条数 {len(data)}，保留 {len(cleaned_items)} 条。输出：{output_path}")


def process_folder(input_dir: str, output_dir: str):
    """
    遍历文件夹，处理所有以 '_filter_rule_base.json' 结尾的文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith("_filter_rule_based.json"):
            continue

        input_path = os.path.join(input_dir, fname)
        # 输出文件名可加一个后缀，例如 _llm_cleaned.json
        base_name, _ = os.path.splitext(fname)
        output_fname = base_name + "_llm_cleaned.json"
        output_path = os.path.join(output_dir, output_fname)

        process_single_file(input_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="使用 LLM (如 gpt-4.1 / gpt-5) 对口腔医学教材 JSON 做相关性判断与清洗。"
    )
    parser.add_argument("--input_dir", type=str, default="/home/jinghao/projects/OralGPT-Agent/Corpus/step_2_rule_filter", help="输入文件夹路径（包含 *_filter_rule_base.json）")
    parser.add_argument("--output_dir", type=str, default="/home/jinghao/projects/OralGPT-Agent/Corpus/step_3_llm_refine", help="输出文件夹路径（保存清洗后的 JSON）")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="LLM 模型名称，例如 gpt-4.1 或未来的 gpt-5")

    args = parser.parse_args()
    global MODEL_NAME
    MODEL_NAME = args.model

    process_folder(args.input_dir, args.output_dir)
