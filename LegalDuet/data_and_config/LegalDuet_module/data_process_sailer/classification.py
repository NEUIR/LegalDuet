import json
from collections import defaultdict
from tqdm import tqdm

input_file_path = '../../../../outside_data/Law_Case.jsonl'
output_file_path = 'frequent_accu_law_combinations.json'

accu_law_counts = defaultdict(int)

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="Processing"):
        data = json.loads(line)
        accu = data['accu']
        law = data['law']
        combo = (accu, law)
        accu_law_counts[combo] += 1

# 过滤出出现次数大于2000次的组合，并将键转换为字符串格式
frequent_combinations = {f"{combo[0]}_{combo[1]}": count for combo, count in accu_law_counts.items() if count > 2000}

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(frequent_combinations, outfile, ensure_ascii=False, indent=4)

print(f"频繁出现的accu和law组合已保存到 {output_file_path}")
