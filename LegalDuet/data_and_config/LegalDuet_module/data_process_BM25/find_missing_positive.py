# import json

# # 输入文件路径
# input_file_path = 'Law_Case_samples_ids.jsonl'
# output_file_path = 'missing_positive_samples.txt'
# vectorized_law_case_file_path = 'vectorized_Law_Case.jsonl'

# # 先加载所有数据到内存中，以便后续查找
# print("加载 vectorized_Law_Case.jsonl 文件...")
# with open(vectorized_law_case_file_path, 'r', encoding='utf-8') as infile:
#     law_cases = {json.loads(line)['id']: json.loads(line) for line in infile}

# missing_positive_samples = []

# print("开始遍历 Law_Case_samples_ids.jsonl 文件...")
# with open(input_file_path, 'r', encoding='utf-8') as infile:
#     for line in infile:
#         data = json.loads(line)
#         if data['positive_sample'] is None:
#             case_id = data['id']
#             case_info = law_cases.get(case_id, {})
#             law = case_info.get('law', 'Unknown')
#             accu = case_info.get('accu', 'Unknown')
#             missing_positive_samples.append({
#                 'id': case_id,
#                 'law': law,
#                 'accu': accu
#             })

# # 将结果写入到 txt 文件
# print("保存没有正样本的 ID 和对应的 law、accu 到文件...")
# with open(output_file_path, 'w', encoding='utf-8') as outfile:
#     for item in missing_positive_samples:
#         outfile.write(f"ID: {item['id']}, Law: {item['law']}, Accu: {item['accu']}\n")

# print(f"缺少正样本的 ID 已保存到 {output_file_path}")
import json

input_file_path = 'Law_Case_samples_ids_updated.jsonl'
output_file_path = 'missing_positive_samples.txt'
vectorized_law_case_file_path = 'vectorized_Law_Case.jsonl'

print("加载 vectorized_Law_Case.jsonl 文件...")
with open(vectorized_law_case_file_path, 'r', encoding='utf-8') as infile:
    law_cases = {json.loads(line)['id']: json.loads(line) for line in infile}

missing_positive_samples = []

print("开始遍历 Law_Case_samples_ids.jsonl 文件...")
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        data = json.loads(line)
        if data['positive_sample'] is None:
            case_id = data['id']
            case_info = law_cases.get(case_id, {})
            law = case_info.get('law', 'Unknown')
            accu = case_info.get('accu', 'Unknown')
            missing_positive_samples.append({
                'id': case_id,
                'law': law,
                'accu': accu
            })

print("保存没有正样本的 ID 和对应的 law、accu 到文件...")
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for item in missing_positive_samples:
        outfile.write(f"ID: {item['id']}, Law: {item['law']}, Accu: {item['accu']}\n")

print(f"缺少正样本的 ID 已保存到 {output_file_path}")
