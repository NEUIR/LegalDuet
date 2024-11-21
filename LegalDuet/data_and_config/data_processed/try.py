import pickle as pk
import numpy as np
import json

# 定义文件路径和要查看的样本数量
file_path = '../data_processed/train_processed_bert.pkl'
num_samples_to_view = 1

# 设置numpy打印选项以避免省略号
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 读取数据文件
with open(file_path, 'rb') as f:
    data_dict = pk.load(f)

# 获取各个数据列表
fact_lists = data_dict['fact_list']
law_label_lists = data_dict['law_label_lists']
accu_label_lists = data_dict['accu_label_lists']
term_lists = data_dict['term_lists']

# 打印前几个样本
for i in range(num_samples_to_view):
    print(f"Sample {i + 1}:")
    print("Fact list (index matrix):")
    print(np.array(fact_lists[i]))
    print("Law labels:", law_label_lists[i])
    print("Accusation labels:", accu_label_lists[i])
    print("Term:", term_lists[i])
    print("\n")

print(f"Total samples in the dataset: {len(fact_lists)}")

# 如果你有原始的 fact_cut 数据，可以对照原始文本和处理后的索引进行检查
# 比如：
with open('../data/train_cs.json', 'r', encoding= 'utf-8') as f:
    for idx, line in enumerate(f.readlines()):
        if idx < num_samples_to_view:
            line = json.loads(line)
            fact = line['fact_cut']
            print(f"Original Fact {idx + 1}:", fact)
            print("\n")

