import json
import random
from tqdm import tqdm

input_file_path = 'Processed_Legal_Ground_Data_with_Hard_Negative.jsonl'
train_file_path = 'train_data.jsonl'
validation_file_path = 'validation_data.jsonl'

print("读取数据...")
data_list = []
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="加载数据", unit=" 行"):
        data = json.loads(line)
        data_list.append(data)

print("打乱数据...")
random.shuffle(data_list)

total_samples = len(data_list)
train_size = int(total_samples * 0.9)
validation_size = total_samples - train_size

train_data = data_list[:train_size]
validation_data = data_list[train_size:]

print(f"总样本数：{total_samples}")
print(f"训练集样本数：{len(train_data)}")
print(f"验证集样本数：{len(validation_data)}")

print("保存训练集...")
with open(train_file_path, 'w', encoding='utf-8') as train_file:
    for data in tqdm(train_data, desc="写入训练集", unit=" 行"):
        train_file.write(json.dumps(data, ensure_ascii=False) + '\n')

print("保存验证集...")
with open(validation_file_path, 'w', encoding='utf-8') as validation_file:
    for data in tqdm(validation_data, desc="写入验证集", unit=" 行"):
        validation_file.write(json.dumps(data, ensure_ascii=False) + '\n')

print("数据集划分完成！")
