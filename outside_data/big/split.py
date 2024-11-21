import json
import random
from tqdm import tqdm

def split_train_validation_and_write(json_file, output_train_file, output_validation_file, train_ratio=0.9):
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 将每一行解析为JSON对象，使用 tqdm 显示进度
    data = [json.loads(line.strip()) for line in tqdm(lines, desc="Loading JSON lines")]

    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    validation_size = total_samples - train_size

    # 随机打乱数据
    random.shuffle(data)

    # 划分数据集
    train_data = data[:train_size]
    validation_data = data[train_size:]

    # 将训练集写入文件，使用 tqdm 显示进度
    with open(output_train_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(train_data, desc="Writing training data"):
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

    # 将验证集写入文件，使用 tqdm 显示进度
    with open(output_validation_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(validation_data, desc="Writing validation data"):
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')

# JSON文件路径
json_file = 'train.json'
output_train_file = 'data_train.json'
output_validation_file = 'data_valid.json'

# 划分训练集和验证集，并写入文件
split_train_validation_and_write(json_file, output_train_file, output_validation_file)

print(f"训练集写入文件：{output_train_file}")
print(f"验证集写入文件：{output_validation_file}")
