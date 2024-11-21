import faiss
import numpy as np
import json

cpu_index = faiss.read_index("law_case_index/index")  

gpu_resources = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)  

with open('vectorized_Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    law_cases = {json.loads(line)['id']: json.loads(line) for line in infile}

first_case_id = 0 
first_case = law_cases[first_case_id]

query_vector = np.array(first_case['vector'], dtype=np.float32)
true_law = first_case['law']
true_accu = first_case['accu']

k = 5
D, I = gpu_index.search(np.array([query_vector]), k=k)

results = []
query_info = {
    "Query Sample ID": first_case_id,
    "Query Law": true_law,
    "Query Accu": true_accu,
    "Top-5 similar samples": []
}

for idx in I[0]:
    similar_case = law_cases[idx]
    result_entry = {
        "Sample ID": similar_case['id'],
        "Law": similar_case['law'],
        "Accu": similar_case['accu'],
        "Matches Query": similar_case['law'] == true_law and similar_case['accu'] == true_accu,
        "Distance": float(D[0][list(I[0]).index(idx)])  
    }
    query_info["Top-5 similar samples"].append(result_entry)

output_file_path = 'top_5_similar_samples.json'
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(query_info, outfile, ensure_ascii=False, indent=4)

print(f"前五个相似样本的信息已保存到 {output_file_path}")
