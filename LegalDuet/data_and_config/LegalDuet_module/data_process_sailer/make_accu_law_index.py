import os
import json

# 定义索引目录
index_dir = 'accu_law'

# 如果目录不存在，则创建
os.makedirs(index_dir, exist_ok=True)

with open('vectorized_Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    for line in infile:
        sample = json.loads(line)
        sample_id = sample['id']
        law = sample['law']
        accu = sample['accu']
        
        filename = f'{law}_{accu}.txt'
        file_path = os.path.join(index_dir, filename)
        
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f"{sample_id}\n")

print(f"索引建立完成，存放在目录: {index_dir}")
