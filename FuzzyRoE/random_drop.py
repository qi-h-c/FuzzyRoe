import json
from tqdm import tqdm

with open('/mnt/42_store/wjr/llava_v1_5_mix665k.json.1', 'r') as f:
    source_datas = json.load(f)

target_datas = []

for idx, data in enumerate(source_datas):
    if idx % 100 < 50:
        target_datas.append(data)
        
print(len(target_datas))

with open('/mnt/42_store/wjr/llava_v1_5_mix665k_2.json', 'w') as f:
    json.dump(target_datas, f, indent=4)
