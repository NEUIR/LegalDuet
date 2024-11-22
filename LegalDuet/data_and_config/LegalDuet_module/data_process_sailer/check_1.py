import faiss
import numpy as np
import json

cpu_index = faiss.read_index("law_case_index/index")  # 替换为实际的law_case索引文件路径

gpu_resources = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)  # 将索引转移到 GPU，0表示GPU ID

with open('vectorized_Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    law_cases = {json.loads(line)['id']: json.loads(line) for line in infile}

# 选择第一个样本作为查询
first_case_id = 0  # 假设第一个样本的ID是0
first_case = law_cases[first_case_id]

query_vector = np.array(first_case['vector'], dtype=np.float32)  # 从 'vector' 字段加载向量
true_law = first_case['law']
true_accu = first_case['accu']

# 初始化检索范围
k = 5
step = 5
found = False

results = []
query_info = {
    "Query Sample ID": first_case_id,
    "Query Law": true_law,
    "Query Accu": true_accu,
    "Top similar samples": []
}

# 循环查找直到找到 law 不是 20 或 accu 不是 56 的样本
while not found:
    D, I = gpu_index.search(np.array([query_vector]), k=k)
    
    for idx in I[0]:
        similar_case = law_cases[idx]
        result_entry = {
            "Sample ID": similar_case['id'],
            "Law": similar_case['law'],
            "Accu": similar_case['accu'],
            "Matches Query": similar_case['law'] == true_law and similar_case['accu'] == true_accu,
            "Distance": float(D[0][list(I[0]).index(idx)])  # 转换为标准 float 类型
        }
        query_info["Top similar samples"].append(result_entry)
        
        # 检查条件：找到 law 不是 20 或 accu 不是 56 的样本
        if similar_case['law'] != true_law or similar_case['accu'] != true_accu:
            found = True
            break
    
    # 扩大查找范围
    k += step

output_file_path = 'top_similar_samples_until_condition.json'
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(query_info, outfile, ensure_ascii=False, indent=4)

print(f"相似样本的信息已保存到 {output_file_path}")
