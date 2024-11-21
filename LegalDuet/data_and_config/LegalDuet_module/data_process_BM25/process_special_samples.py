import os
import json
from tqdm import tqdm

input_dir = 'missing_samples_by_accu_law'
input_file_path = 'Law_Case_samples_ids.jsonl'
output_file_path = 'Law_Case_samples_ids_updated.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as infile:
    samples = {json.loads(line)['id']: json.loads(line) for line in infile}

for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            if len(lines) == 2:  # 如果文件中只有两个样本
                id1 = int(lines[0].split(',')[0].split(':')[1].strip())
                id2 = int(lines[1].split(',')[0].split(':')[1].strip())

                if id1 in samples and id2 in samples:
                    samples[id1]['positive_sample'] = id2
                    samples[id2]['positive_sample'] = id1
                    print(f"已将 {id1} 和 {id2} 互为正样本处理")
            elif len(lines) == 1:  # 如果文件中只有一个样本
                id1 = int(lines[0].split(',')[0].split(':')[1].strip())
                if id1 in samples:
                    samples[id1]['positive_sample'] = id1  # 自己设置为自己的正样本
                    print(f"已将 {id1} 设置为自己的正样本")

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for sample in tqdm(samples.values(), desc="保存更新后的样本"):
        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"更新后的样本结果已保存到 {output_file_path}")
