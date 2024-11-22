import faiss
import numpy as np
import json
from tqdm import tqdm

cpu_index = faiss.read_index("law_case_index/index")  

with open('vectorized_Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    law_cases = {json.loads(line)['id']: json.loads(line) for line in infile}

input_file_path_gpu = 'query_samples_result_gpu_2.jsonl'
output_file_path_cpu = 'query_samples_result_cpu.jsonl'

search_limit_cpu = 100000

with open(input_file_path_gpu, 'r', encoding='utf-8') as infile:
    samples = [json.loads(line) for line in infile]

with open(output_file_path_cpu, 'w', encoding='utf-8') as outfile:
    for result_data in tqdm(samples, desc="Processing samples with CPU"):
        sample_id = result_data['query_id']
        first_case = law_cases[sample_id]
        query_vector = np.array(first_case['vector'], dtype=np.float32)
        true_law = first_case['law']
        true_accu = first_case['accu']

        k = 1000  
        step = 1000
        found_positive = bool(result_data['positive_sample']) 
        negative_samples = result_data['negative_samples'] 

        # 如果正样本或负样本未完成，继续处理
        if (len(negative_samples) < 15) or (not found_positive):
            # 查找正样本
            while not found_positive and k <= search_limit_cpu:
                D, I = cpu_index.search(np.array([query_vector]), k=k)

                for idx in I[0]:
                    similar_case = law_cases[idx]

                    # 正样本条件：law 和 accu 都匹配，但不是查询样本本身
                    if similar_case['law'] == true_law and similar_case['accu'] == true_accu and idx != sample_id:
                        result_data["positive_sample"] = {
                            "sample_id": similar_case['id'],
                            "law": similar_case['law'],
                            "accu": similar_case['accu']
                        }
                        found_positive = True
                        break

                if k >= 100000:
                    result_data["positive_sample"] = {
                        "sample_id": sample_id,
                        "law": true_law,
                        "accu": true_accu
                    }
                    found_positive = True  # 直接退出搜索
                    print(sample_id)
                    break
                # 扩大查找范围
                k += step

            # 查找负样本
            while len(negative_samples) < 15 and k <= search_limit_cpu:
                D, I = cpu_index.search(np.array([query_vector]), k=k)

                for idx in I[0]:
                    similar_case = law_cases[idx]

                    # 负样本条件：law 或 accu 不匹配，且不是正样本和查询样本本身
                    if (similar_case['law'] != true_law or similar_case['accu'] != true_accu) and idx != sample_id:
                        if similar_case['id'] not in [sample['sample_id'] for sample in negative_samples]:
                            negative_samples.append({
                                "sample_id": similar_case['id'],
                                "law": similar_case['law'],
                                "accu": similar_case['accu']
                            })

                    # 如果已经找到15个负样本，跳出循环
                    if len(negative_samples) >= 15:
                        break

                # 扩大查找范围
                k += step

            # 更新负样本列表
            result_data["negative_samples"] = negative_samples

        # 无论是否进行了补充处理，都将结果保存到新文件中
        outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')

print(f"CPU处理完成，结果已保存到 {output_file_path_cpu}")
