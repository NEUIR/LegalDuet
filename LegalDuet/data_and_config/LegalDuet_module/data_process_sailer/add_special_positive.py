import os
import json

index_dir = 'accu_law'
input_file_path_gpu = 'query_samples_result_gpu.jsonl'
output_file_path_cpu = 'query_samples_result_gpu_2.jsonl'

with open(input_file_path_gpu, 'r', encoding='utf-8') as infile:
    samples = {json.loads(line)['query_id']: json.loads(line) for line in infile}

# 遍历索引目录，找到只有两个样本的文件
for filename in os.listdir(index_dir):
    file_path = os.path.join(index_dir, filename)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        ids = file.read().splitlines()
    
    # 如果文件中只有两个样本，进行处理
    if len(ids) == 2:
        id1, id2 = int(ids[0]), int(ids[1])
        
        # 更新第一个样本的正样本
        if id1 in samples:
            samples[id1]["positive_sample"] = {
                "sample_id": id2,
                "law": samples[id2]['query_law'],
                "accu": samples[id2]['query_accu']
            }
        print(id1)
        # 更新第二个样本的正样本
        if id2 in samples:
            samples[id2]["positive_sample"] = {
                "sample_id": id1,
                "law": samples[id1]['query_law'],
                "accu": samples[id1]['query_accu']
            }
        print(id2)
        
with open(output_file_path_cpu, 'w', encoding='utf-8') as outfile:
    for sample in samples.values():
        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"处理完成，结果已保存到 {output_file_path_cpu}")
