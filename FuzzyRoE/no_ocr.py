import json
import os

input_path = "/mnt/42_store/wjr/llava_v1_5_mix665k_2.json"
output_path = "/mnt/42_store/wjr/llava_v1_5_mix665k_no_ocr.json"

with open(input_path, "r", encoding="utf-8") as fin:
    data = json.load(fin)

filtered = []
for item in data:
    img_path = item.get("image", "")
    # 过滤掉 OCR‑VQA 相关条目
    # 注意可能存在大小写或下划线等差异，可以根据实际情况再加几个 startswith 条件
    if not (img_path.startswith("ocr_vqa/")):
        filtered.append(item)

# 保存过滤后的文件
with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(filtered, fout, ensure_ascii=False, indent=2)

print(f"原始条目数：{len(data)}, 过滤后条目数：{len(filtered)}")
