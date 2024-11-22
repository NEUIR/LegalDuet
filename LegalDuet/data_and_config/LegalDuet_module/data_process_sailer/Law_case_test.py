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
    # 先加载所有数据到内存中，以便后续查找
    law_cases = [json.loads(line) for line in infile]

print("开始处理样本...")
for current_case in tqdm(law_cases, desc="Processing"):
    query_vector = np.array(current_case['vector'], dtype=np.float32)  # 从 'vector' 字段加载向量
    
    # 获取当前样本的accu和law
    true_accu = current_case['accu']
    true_law = current_case['law']
    current_id = current_case['id']

    print(f"处理样本 ID: {current_id}, accu: {true_accu}, law: {true_law}")

    # 初始查找范围
    k = 100
    steps = 100  # 每次增加的查找范围
    max_k = 2000  # 设置一个最大查找范围

    positive_sample = None
    negative_samples = []
    
    while len(negative_samples) < 15 and k <= max_k:
        print(f"正在搜索，当前 k = {k}...")
        # 在GPU上的law_case索引中进行检索
        D, I = gpu_index.search(np.array([query_vector]), k=k)  # 检索前k个最相似的样本

        for idx in I[0]:
            similar_case = law_cases[idx]
            if similar_case['id'] == current_id:
                continue  # 排除自身
            
            # 筛选正样本：accu和law字段均相同
            if similar_case['accu'] == true_accu and similar_case['law'] == true_law:
                if positive_sample is None:
                    positive_sample = similar_case['id']
            
            # 筛选负样本：accu和law字段至少有一个不同
            if similar_case['accu'] != true_accu or similar_case['law'] != true_law:
                if similar_case['id'] not in negative_samples:
                    negative_samples.append(similar_case['id'])

            # 如果已经找到足够的负样本，可以提前结束
            if len(negative_samples) >= 15:
                break

        # 如果还没有足够的负样本，扩大查找范围
        k += steps

    result_entry = {
        'id': current_id,
        'positive_sample': positive_sample,
        'negative_samples': negative_samples[:15]  # 只保留前15个负样本
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
