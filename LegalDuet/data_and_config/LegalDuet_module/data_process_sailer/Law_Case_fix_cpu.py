import json
from tqdm import tqdm
import faiss
import numpy as np

cpu_index = faiss.read_index("law_case_index/index")  # 替换为实际的law_case索引文件路径

input_file_path = 'query_samples_result_cpu.jsonl'
output_file_path = 'updated_query_samples_result.jsonl'
vectorized_law_case_file = 'vectorized_Law_Case.jsonl'  # 包含样本向量的文件

with open(vectorized_law_case_file, 'r', encoding='utf-8') as infile:
    law_cases = {json.loads(line)['id']: json.loads(line)['vector'] for line in infile}

with open(input_file_path, 'r', encoding='utf-8') as infile:
    query_samples_data = {json.loads(line)['query_id']: json.loads(line) for line in infile}

# 初始化检索上限和批处理设置
search_limit = 702000  # 新的更大检索范围
batch_size = 10000  # 每处理10000个样本保存一次

updated_results = []
batch_counter = 0  

def search_for_samples(sample_id, query_vector, query_law, query_accu, search_limit, step=100000):
    """根据查询向量进行正样本和负样本检索"""
    k = 100000
    found_positive = False
    negative_samples = []
    positive_sample = None

    # 查找正样本
    while not found_positive and k <= search_limit:
        D, I = cpu_index.search(np.array([query_vector]), k=k)

        for idx in I[0]:
            similar_case_id = int(idx)
            # 从 query_samples_data 中获取 law 和 accu
            similar_case_data = query_samples_data.get(similar_case_id, None)
            if similar_case_data is not None:
                # 正样本条件：law 和 accu 都匹配，但不是查询样本本身
                if similar_case_data['query_law'] == query_law and similar_case_data['query_accu'] == query_accu and similar_case_id != sample_id:
                    positive_sample = {
                        "sample_id": similar_case_data['query_id'],
                        "law": similar_case_data['query_law'],
                        "accu": similar_case_data['query_accu']
                    }
                    found_positive = True
                    break

        k += step

    # 查找负样本
    k = 100000  # 重新初始化检索范围
    while len(negative_samples) < 15 and k <= search_limit:
        D, I = cpu_index.search(np.array([query_vector]), k=k)

        for idx in I[0]:
            similar_case_id = int(idx)
            # 从 query_samples_data 中获取 law 和 accu
            similar_case_data = query_samples_data.get(similar_case_id, None)
            if similar_case_data is not None:
                # 负样本条件：law 或 accu 不匹配，且不是正样本和查询样本本身
                if (similar_case_data['query_law'] != query_law or similar_case_data['query_accu'] != query_accu) and similar_case_id != sample_id:
                    if similar_case_data['query_id'] not in [sample['sample_id'] for sample in negative_samples]:
                        negative_samples.append({
                            "sample_id": similar_case_data['query_id'],
                            "law": similar_case_data['query_law'],
                            "accu": similar_case_data['query_accu']
                        })
            if len(negative_samples) >= 15:
                break

        k += step

    return positive_sample, negative_samples


# 遍历 query_samples_result_cpu.jsonl 文件
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing samples"):
        result_data = json.loads(line)
        sample_id = result_data['query_id']
        query_law = result_data['query_law']
        query_accu = result_data['query_accu']

        # 获取query样本向量
        if sample_id in law_cases:
            query_vector = np.array(law_cases[sample_id], dtype=np.float32)
        else:
            print(f"向量不存在，样本ID: {sample_id}")
            continue

        # 如果正样本与自身一致，说明之前检索范围超过100000
        if result_data['positive_sample']['sample_id'] == sample_id:
            print(f"正样本与自身一致，重新检索样本ID: {sample_id}")
            positive_sample, negative_samples = search_for_samples(sample_id, query_vector, query_law, query_accu, search_limit)

            # 更新正样本和负样本信息
            result_data['positive_sample'] = positive_sample
            result_data['negative_samples'] = negative_samples

        # 如果负样本不足15个，说明可能需要扩大检索范围
        if len(result_data['negative_samples']) < 15:
            print(f"负样本不足，重新检索样本ID: {sample_id}")
            positive_sample, negative_samples = search_for_samples(sample_id, query_vector, query_law, query_accu, search_limit)

            # 更新负样本信息
            result_data['negative_samples'] = negative_samples

        # 将更新后的结果添加到列表
        updated_results.append(result_data)
        batch_counter += 1

        # 每处理10000个样本保存一次
        if batch_counter >= batch_size:
            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                for result in updated_results:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            updated_results = []  # 清空结果列表
            batch_counter = 0  # 重置计数器

    # 处理最后一批未保存的结果
    if updated_results:
        with open(output_file_path, 'a', encoding='utf-8') as outfile:
            for result in updated_results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"更新后的样本已保存到 {output_file_path}")
