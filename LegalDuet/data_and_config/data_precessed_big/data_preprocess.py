import json
from tqdm import tqdm

target_accusations = {
    111: "故意伤害",
    108: "故意杀人",
    110: "盗窃",
    12: "抢劫",
    1: "寻衅滋事",
    55: "诈骗"
}

input_file_path = 'train_cs_bert.json'  # 使用处理后的分词数据集
filtered_file_path = 'filtered_Law_Case.jsonl'  # 筛选后的输出文件

accu_count = {accu: 0 for accu in target_accusations.keys()}

with open(input_file_path, 'r', encoding='utf-8') as infile, open(filtered_file_path, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Filtering and Counting"):
        data = json.loads(line)
        accu_label = data['accu'] 

        if accu_label in target_accusations:
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            accu_count[accu_label] += 1

print("各罪行的数量统计：")
for accu_label, count in accu_count.items():
    print(f"{target_accusations[accu_label]} ({accu_label}): {count}条")

print(f"筛选后的数据已保存到 {filtered_file_path}")
