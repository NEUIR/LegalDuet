import faiss
import numpy as np
import json
from tqdm import tqdm

print("加载 Faiss 索引...")
cpu_index = faiss.read_index("law_case_index/index")  

print("初始化 GPU 资源...")
gpu_resources = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index) 

input_file_path = 'vectorized_Law_Case.jsonl'
output_file_path = 'Law_Case_samples_ids.jsonl'

results = []
batch_size = 1000  
batch_counter = 0 

print("加载 JSONL 文件中的向量数据...")
with open(input_file_path, 'r', encoding='utf-8') as infile:
    law_cases = [json.loads(line) for line in infile]

print("开始处理样本...")
for current_case in tqdm(law_cases, desc="Processing"):
    query_vector = np.array(current_case['vector'], dtype=np.float32) 
    
    true_accu = current_case['accu']
    true_law = current_case['law']
    current_id = current_case['id']

    print(f"处理样本 ID: {current_id}, accu: {true_accu}, law: {true_law}")

    k = 100
    steps = 100  
    max_k = 2000 

    positive_sample = None
    negative_samples = []
    
    while len(negative_samples) < 15 and k <= max_k:
        print(f"正在搜索，当前 k = {k}...")
        D, I = gpu_index.search(np.array([query_vector]), k=k)  

        for idx in I[0]:
            similar_case = law_cases[idx]
            if similar_case['id'] == current_id:
                continue  
            
            if similar_case['accu'] == true_accu and similar_case['law'] == true_law:
                if positive_sample is None:
                    positive_sample = similar_case['id']
            
            if similar_case['accu'] != true_accu or similar_case['law'] != true_law:
                if similar_case['id'] not in negative_samples:
                    negative_samples.append(similar_case['id'])

            if len(negative_samples) >= 15:
                break

        k += steps

    result_entry = {
        'id': current_id,
        'positive_sample': positive_sample,
        'negative_samples': negative_samples[:15]  
    }
    results.append(result_entry)

    batch_counter += 1

    if batch_counter >= batch_size:
        print(f"已处理 {batch_counter} 个样本，保存中间结果...")
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

print(f"law_case_reasoning 样本结果已保存到 {output_file_path}")
