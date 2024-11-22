import faiss
import os
import numpy as np
import json
from tqdm import tqdm

index_dir = 'sub_datasets'
global_index_path = 'law_case_index/index' 

gpu_resources_0 = faiss.StandardGpuResources()
gpu_resources_1 = faiss.StandardGpuResources()
gpu_resources_3 = faiss.StandardGpuResources()

custom_indices = {}
gpu_mapping = {
    "56_20": 0,
    "55_18": 0,
    "31_6": 0,
    "10_49": 0,
    "8_13": 0,
    "54_47": 0,
    "46_13": 0,
    "23_41": 0,
    "1_53": 0,
    "34_26": 0,
    "18_40": 1,
    "9_2": 1,
    "40_29": 1,
    "4_21": 1,
    "16_16": 1,
    "0_57": 1,
    "6_36": 1,
    "53_24": 1,
    "43_15": 1,
    "15_37": 1,
    "60_11": 3,
    "37_3": 3,
    "61_55": 3,
    "41_5": 3,
    "25_25": 3,
    "28_39": 3,
    "2_25": 3,
    "24_50": 3,
    "36_54": 3
}

print("加载索引到 GPU...")
for combo in os.listdir(index_dir):
    if combo.endswith('_index'):
        combo_key = combo.replace('_index', '')
        index_path = os.path.join(index_dir, combo, 'index')  # 假设索引文件名为 'index'
        print(f"加载索引: {combo_key} -> {index_path}")
        cpu_index = faiss.read_index(index_path)
        gpu_id = gpu_mapping.get(combo_key, 0)  

        if gpu_id == 0:
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources_0, gpu_id, cpu_index)
        elif gpu_id == 1:
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources_1, gpu_id, cpu_index)
        elif gpu_id == 3:
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources_3, gpu_id, cpu_index)
        
        custom_indices[combo_key] = gpu_index

print("加载全局索引到 GPU 0...")
cpu_global_index = faiss.read_index(global_index_path)
global_index = faiss.index_cpu_to_gpu(gpu_resources_0, 0, cpu_global_index)

print("加载 JSONL 文件中的向量数据...")
with open('vectorized_Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    law_cases = [json.loads(line) for line in tqdm(infile, desc="Loading data")]

law_cases_dict = {case['id']: case for case in law_cases}

def search(query_vector, true_accu, true_law, k):
    combo_key = f"{true_accu}_{true_law}"
    # print(f"正在搜索: {combo_key} (top-{k})")
    
    if combo_key in custom_indices:
        index = custom_indices[combo_key]
    else:
        index = global_index

    D, I = index.search(np.array([query_vector]), k=k)
    return D, I

output_file_path = 'Law_Case_samples_ids.jsonl'

results = []
batch_size = 1000  
batch_counter = 0 

print("开始处理样本...")
for current_case in tqdm(law_cases, desc="Processing"):
    query_vector = np.array(current_case['vector'], dtype=np.float32)
    true_accu = current_case['accu']
    true_law = current_case['law']
    current_id = current_case['id']
        
    if current_id < 100:
        print(f"正在处理样本: {current_id}")
    if current_id % 10000 == 0:
        print(f"已处理样本数: {current_id}")

    # 初始化搜索范围
    k = 100
    steps = 100  # 每次增加的范围
    max_k = 2000  # 最大搜索范围

    positive_sample = None
    negative_samples = []

    while len(negative_samples) < 15 and k <= max_k:
        D, I = search(query_vector, true_accu, true_law, k=k)

        for idx in I[0]:
            if idx == int(current_id):  # 确保比较的是整数形式的 ID
                continue

            similar_case = law_cases_dict.get(idx)

            if similar_case:
                if similar_case['accu'] == true_accu and similar_case['law'] == true_law:
                    if positive_sample is None:
                        positive_sample = similar_case['id']
                else:
                    if len(negative_samples) < 15:
                        negative_samples.append(similar_case['id'])

        # 如果负样本不足，扩大搜索范围
        if len(negative_samples) < 15:
            # print(f"扩大搜索范围: k = {k + steps}")
            k += steps

    result_entry = {
        'id': current_id,
        'positive_sample': positive_sample,
        'negative_samples': negative_samples[:15]
    }
    results.append(result_entry)

    batch_counter += 1

    # 每处理 batch_size 个样本就保存一次
    if batch_counter >= batch_size:
        print(f"保存中间结果... (已处理 {batch_counter} 个样本)")
        with open(output_file_path, 'a', encoding='utf-8') as outfile:
            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
        results = [] 
        batch_counter = 0  

if results:
    print("保存最后一批结果...")
    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"检索结果已保存到 {output_file_path}")
