import json
from tqdm import tqdm

# 目标罪名类别
target_accusations = {
    111: "故意伤害",
    108: "故意杀人",
    110: "盗窃",
    12: "抢劫",
    1: "寻衅滋事",
    55: "诈骗"
}

# 文件路径
input_file_path = '/data1/xubuqiang/outside_data/test_cs_bert.json'  # 处理后的数据集
filtered_file_path = 'filtered_Law_Case_test.jsonl'  # 筛选后的输出文件
all_filtered_file_path = 'filtered_Law_Case_test_all.jsonl'  # 直接保留所有符合条件的数据

# 统计数量 & 存储筛选数据
accu_count = {accu: 0 for accu in target_accusations.keys()}
filtered_data = []

# 读取并筛选数据
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="Filtering Cases"):
        data = json.loads(line)
        accu_label = data['accu']
        
        if accu_label in target_accusations:
            filtered_data.append(data)
            accu_count[accu_label] += 1

# 保存所有筛选后的数据（不再做随机抽样）
with open(filtered_file_path, 'w', encoding='utf-8') as outfile:
    for item in filtered_data:
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存完整筛选数据到另一个文件（与 `filtered_file_path` 内容相同）
with open(all_filtered_file_path, 'w', encoding='utf-8') as outfile:
    for item in filtered_data:
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

# 输出统计信息
print("各罪行的数量统计：")
for accu_label, count in accu_count.items():
    print(f"{target_accusations[accu_label]} ({accu_label}): {count} 条")

print(f"筛选后的数据已保存到 {filtered_file_path}")
print(f"所有符合条件的数据已保存到 {all_filtered_file_path}")
