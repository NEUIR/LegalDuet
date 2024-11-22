import faiss
import numpy as np
import json
from tqdm import tqdm

# 加载全局索引 (CPU)
cpu_index = faiss.read_index("law_case_index/index")  

with open('vectorized_Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    law_cases = {json.loads(line)['id']: json.loads(line) for line in infile}

input_file_path = 'Law_Case_samples_ids_updated.jsonl'
output_file_path = 'Law_Case_samples_ids_final.jsonl'
log_file_path = 'samples_with_large_k.txt'  # 用于记录k达到100000的样本ID

results = []
batch_size = 10000  # 每处理10000个样本就保存一次
batch_counter = 0  # 计数器

with open(input_file_path, 'r', encoding='utf-8') as infile, open(log_file_path, 'w', encoding='utf-8') as log_file:
    for line in tqdm(infile, desc="Processing samples"):
        data = json.loads(line)
        if data['positive_sample'] is None:  # 如果正样本仍然缺失
            case_id = data['id']
            current_case = law_cases[case_id]
            true_law = current_case['law']
            true_accu = current_case['accu']
            query_vector = np.array(current_case['vector'], dtype=np.float32)

            k = 100
            steps = 100
            max_k = 702000

            positive_sample = None

            while positive_sample is None and k <= max_k:
                D, I = cpu_index.search(np.array([query_vector]), k=k)

                for idx in I[0]:
                    if idx == case_id:
                        continue  # 排除自身
                    similar_case = law_cases[idx]
                    if similar_case['accu'] == true_accu and similar_case['law'] == true_law:
                        positive_sample = similar_case['id']
                        break

                k += steps

            if positive_sample is None:
                # 如果k达到max_k仍未找到正样本，将自身设置为正样本，并记录ID
                positive_sample = case_id
                log_message = f"ID: {case_id} reached k={max_k}, set positive_sample to itself.\n"
                log_file.write(log_message)
                print(log_message)

            # 更新正样本ID
            data['positive_sample'] = positive_sample

        results.append(data)
        batch_counter += 1

        if batch_counter >= batch_size:
            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                for result in results:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            results = []  # 清空结果列表
            batch_counter = 0  # 重置计数器

# 处理最后一批未保存的结果
if results:
    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"最终更新后的样本结果已保存到 {output_file_path}")
print(f"所有达到 k=100000 的样本ID已记录在 {log_file_path}")
