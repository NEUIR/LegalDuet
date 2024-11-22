import json
from collections import defaultdict
from tqdm import tqdm

input_file_path = 'Processed_Law_Case_Data.jsonl'
output_file_path = 'Label_Statistics.json'

label_counts = defaultdict(int)

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="统计标签", unit=" 行"):
        data = json.loads(line)
        # 获取 fact_label 和 positive_label
        fact_label = data.get('fact_label')
        
        if fact_label is not None:
            label_counts[fact_label] += 1

sorted_label_counts = dict(sorted(label_counts.items(), key=lambda item: item[0]))

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(sorted_label_counts, outfile, ensure_ascii=False, indent=4)

print(f"标签统计结果已保存到 {output_file_path}")
