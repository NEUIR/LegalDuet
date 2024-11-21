import json
import random

def split_train_validation_test_and_write(json_file, output_train_file, output_validation_file, output_test_file, train_ratio=0.8, validation_ratio=0.1):
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 将每一行解析为 JSON 对象
    data = [json.loads(line.strip()) for line in lines]

    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    validation_size = int(total_samples * validation_ratio)
    test_size = total_samples - train_size - validation_size

    # 随机打乱数据
    random.shuffle(data)

    # 划分数据集
    train_data = data[:train_size]
    validation_data = data[train_size:train_size+validation_size]
    test_data = data[train_size+validation_size:]

    # 将训练集写入文件
    with open(output_train_file, 'w', encoding='utf-8') as f:
        for sample in train_data:
            json.dump(sample, f, ensure_ascii=False)  # 设置 ensure_ascii=False
            f.write('\n')

    # 将验证集写入文件
    with open(output_validation_file, 'w', encoding='utf-8') as f:
        for sample in validation_data:
            json.dump(sample, f, ensure_ascii=False)  # 设置 ensure_ascii=False
            f.write('\n')

    # 将测试集写入文件
    with open(output_test_file, 'w', encoding='utf-8') as f:
        for sample in test_data:
            json.dump(sample, f, ensure_ascii=False)  # 设置 ensure_ascii=False
            f.write('\n')

# JSON 文件路径
json_file = open("../../../outside_data/rest_data.json",'r',encoding='utf-8')
output_train_file = '../../../outside_data/rest_data_train.json'
output_validation_file = '../../../outside_data/rest_data_valid.json'
output_test_file ='../../../outside_data/rest_data_test.json'

# 划分训练集、验证集和测试集，并写入文件
split_train_validation_test_and_write(json_file, output_train_file, output_validation_file, output_test_file)

print(f"训练集写入文件：{output_train_file}")
print(f"验证集写入文件：{output_validation_file}")
print(f"测试集写入文件：{output_test_file}")