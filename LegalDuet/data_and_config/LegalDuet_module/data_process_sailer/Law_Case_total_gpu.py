import faiss
import numpy as np
import json
from tqdm import tqdm

cpu_index = faiss.read_index("law_case_index/index") 
gpu_resources = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 3, cpu_index)  

with open('vectorized_Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    law_cases = {json.loads(line)['id']: json.loads(line) for line in infile}

output_file_path_gpu = 'query_samples_result_gpu.jsonl'
output_file_path_pending_cpu = 'samples_pending_cpu.txt'

search_limit_gpu = 2048
batch_size = 1000

result_data_list = []
pending_samples_ids = []  

for sample_id, first_case in tqdm(law_cases.items(), desc="Processing samples with GPU"):
    query_vector = np.array(first_case['vector'], dtype=np.float32)
    true_law = first_case['law']
    true_accu = first_case['accu']

    k = 1000
    step = 1000
    found_positive = False
    negative_samples = []

    result_data = {
        "query_id": first_case['id'],
        "query_law": true_law,
        "query_accu": true_accu,
        "positive_sample": {},
        "negative_samples": []
    }

    while not found_positive and k <= search_limit_gpu:
        D, I = gpu_index.search(np.array([query_vector]), k=k)
        
        for idx in I[0]:
            similar_case = law_cases[idx]
            
            if similar_case['law'] == true_law and similar_case['accu'] == true_accu and idx != sample_id:
                result_data["positive_sample"] = {
                    "sample_id": similar_case['id'],
                    "law": similar_case['law'],
                    "accu": similar_case['accu']
                }
                found_positive = True
                break
        
        k += step

    k = 1000

    while len(negative_samples) < 15 and k <= search_limit_gpu:
        D, I = gpu_index.search(np.array([query_vector]), k=k)
        
        for idx in I[0]:
            similar_case = law_cases[idx]
            
            if (similar_case['law'] != true_law or similar_case['accu'] != true_accu) and idx != sample_id:
                if similar_case['id'] not in [sample['sample_id'] for sample in negative_samples]:
                    negative_samples.append({
                        "sample_id": similar_case['id'],
                        "law": similar_case['law'],
                        "accu": similar_case['accu']
                    })
            
            if len(negative_samples) >= 15:
                break
        
        k += step

    result_data["negative_samples"] = negative_samples

    if (len(negative_samples) < 15) or (not found_positive):
        pending_samples_ids.append(result_data['query_id'])

    result_data_list.append(result_data)

    if len(result_data_list) % batch_size == 0:
        with open(output_file_path_gpu, 'a', encoding='utf-8') as outfile:
            for result in result_data_list:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
        result_data_list = []  
    
    if len(pending_samples_ids) % batch_size == 0:
        with open(output_file_path_pending_cpu, 'a', encoding='utf-8') as pending_outfile:
            for pending_sample_id in pending_samples_ids:
                pending_outfile.write(f"{pending_sample_id}\n")
        pending_samples_ids = []  

if result_data_list:
    with open(output_file_path_gpu, 'a', encoding='utf-8') as outfile:
        for result in result_data_list:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

if pending_samples_ids:
    with open(output_file_path_pending_cpu, 'a', encoding='utf-8') as pending_outfile:
        for pending_sample_id in pending_samples_ids:
            pending_outfile.write(f"{pending_sample_id}\n")

print(f"GPU处理完成，结果已保存到 {output_file_path_gpu}")
print(f"未完成的样本已保存到 {output_file_path_pending_cpu}，供CPU后续处理")
