import json
import random

input_file_path = "Merged_Law_Ground_Data.jsonl"  # 整合后的数据集路径
train_file_path = "train.jsonl"                   # 训练集保存路径
validation_file_path = "validation.jsonl"         # 验证集保存路径

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

print(f"Loading dataset from {input_file_path}...")
data = load_jsonl(input_file_path)
print(f"Dataset loaded. Total samples: {len(data)}")

print("Shuffling dataset...")
random.seed(42)  
random.shuffle(data)
print("Dataset shuffled.")

validation_size = 5000
validation_data = data[:validation_size]  # 取前 5000 个样本作为验证集
train_data = data[validation_size:]       # 剩余样本作为训练集

print(f"Saving validation dataset to {validation_file_path}...")
save_jsonl(validation_data, validation_file_path)
print(f"Validation dataset saved. Total samples: {len(validation_data)}")

print(f"Saving training dataset to {train_file_path}...")
save_jsonl(train_data, train_file_path)
print(f"Training dataset saved. Total samples: {len(train_data)}")

print("Dataset split and saved successfully.")
