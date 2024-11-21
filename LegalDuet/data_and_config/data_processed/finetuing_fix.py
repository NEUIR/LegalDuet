import pickle as pk
import json
import numpy as np

# 加载 pkl 文件
file_list = ['train']

for file in file_list:
    # 加载 pkl 文件中的数据
    with open(f'{file}_processed_sailer.pkl', 'rb') as pkl_file:
        data = pk.load(pkl_file)

    # 初始化列表，保存转换后的 JSON 数据
    jsonl_data = []

    # 遍历所有样本并添加索引 idx
    for idx, (fact, law_label, accu_label, term) in enumerate(zip(data['fact_list'], data['law_label_lists'], data['accu_label_lists'], data['term_lists'])):
        # 将 numpy 数组转换为列表以便 JSON 化
        fact_list = fact.tolist() if isinstance(fact, np.ndarray) else fact

        # 构建 JSON 数据格式
        sample_data = {
            'idx': idx,  # 为每个样本添加唯一的 idx 字段
            'fact': fact_list,
            'law_label': law_label,
            'accu_label': accu_label,
            'term': term
        }
        
        # 将样本添加到列表中
        jsonl_data.append(sample_data)

        # 如果是第一个样本，则单独保存到一个文件中
        if idx == 0:
            first_sample_output_file = f'{file}_first_sample.json'
            with open(first_sample_output_file, 'w', encoding='utf-8') as first_sample_file:
                json.dump(sample_data, first_sample_file, ensure_ascii=False, indent=4)
            print(f'{file} 数据集第一个样本已保存到 {first_sample_output_file}')

    # 保存为 jsonl 文件
    output_file = f'{file}_processed_sailer.jsonl'
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for sample in jsonl_data:
            jsonl_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f'{file} 数据集已处理并保存为 {output_file}')
