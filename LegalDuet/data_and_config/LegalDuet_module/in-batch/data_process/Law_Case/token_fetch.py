# import json
# from tqdm import tqdm

# # 定义文件路径
# tokens_file_path = 'Law_Case_with_tokens.jsonl'
# samples_file_path = 'Law_Case_samples_ids_final.jsonl'
# output_file_path = 'Processed_Law_Case_Data.jsonl'
# first_sample_output_path = 'First_Sample.json'

# # Step 1: 加载 Law_Case_with_tokens.jsonl 到字典中，包含标签信息
# token_data = {}

# print("加载已分词的数据...")
# with open(tokens_file_path, 'r', encoding='utf-8') as tokens_file:
#     for line in tqdm(tokens_file, desc="加载 tokens", unit=" 行"):
#         data = json.loads(line)
#         sample_id = data['id']
        
#         # 提取 law 和 accu 标签，确保它们是整数，并从 0 开始
#         law = data['law']  # 假设 law 已经编码为 0 到 58 的整数
#         accu = data['accu']  # 假设 accu 已经编码为 0 到 61 的整数
        
#         token_data[sample_id] = {
#             'token_ids': data['token_ids'],
#             'law': law,     # law 标签
#             'accu': accu    # accu 标签
#         }

# # Step 2: 处理 Law_Case_samples_ids_final.jsonl
# print("处理样本数据...")
# with open(samples_file_path, 'r', encoding='utf-8') as samples_file, \
#      open(output_file_path, 'w', encoding='utf-8') as output_file:
    
#     first_sample_processed = False  # 标记是否已处理第一个样本

#     for line in tqdm(samples_file, desc="处理样本", unit=" 行"):
#         sample = json.loads(line)
#         fact_id = sample['id']
#         positive_id = sample['positive_sample']
        
#         # 获取 fact 和 positive 的数据
#         fact_data = token_data.get(fact_id)
#         positive_data = token_data.get(positive_id)
        
#         if fact_data is None:
#             print(f"警告：未找到 fact id {fact_id} 的数据。")
#             continue
#         if positive_data is None:
#             print(f"警告：未找到 positive sample id {positive_id} 的数据。")
#             continue
        
#         # 计算新的标签
#         fact_label = fact_data['law'] * 62 + fact_data['accu']
#         positive_label = positive_data['law'] * 62 + positive_data['accu']
        
#         # 创建新的样本数据，包含新的标签
#         processed_sample = {
#             'id': fact_id,
#             'fact_token_ids': fact_data['token_ids'],
#             'fact_label': fact_label,
#             'positive_id': positive_id,
#             'positive_token_ids': positive_data['token_ids'],
#             'positive_label': positive_label
#         }
        
#         # 将处理后的样本写入输出文件
#         output_file.write(json.dumps(processed_sample, ensure_ascii=False) + '\n')
        
#         # 保存并展示第一个样本的处理结果
#         if not first_sample_processed:
#             with open(first_sample_output_path, 'w', encoding='utf-8') as first_sample_file:
#                 json.dump(processed_sample, first_sample_file, ensure_ascii=False, indent=4)
#             first_sample_processed = True

# print(f"处理后的数据已保存到 {output_file_path}")
# print(f"第一个样本的处理结果已保存到 {first_sample_output_path}")

# # 展示第一个样本的处理结果
# with open(first_sample_output_path, 'r', encoding='utf-8') as f:
#     first_sample = json.load(f)
#     print("第一个样本的处理结果：")
#     print(json.dumps(first_sample, ensure_ascii=False, indent=4))


import json
from tqdm import tqdm

tokens_file_path = 'Law_Case_with_tokens.jsonl'
samples_file_path = 'converted_query_samples_result.jsonl'
output_file_path = 'Processed_Law_Case_Data_with_Hard_Negative.jsonl'
first_sample_output_path = 'First_Sample_with_Hard_Negative.json'

token_data = {}

print("加载已分词的数据...")
with open(tokens_file_path, 'r', encoding='utf-8') as tokens_file:
    for line in tqdm(tokens_file, desc="加载 tokens", unit=" 行"):
        data = json.loads(line)
        sample_id = data['id']
        
        law = data['law']  
        accu = data['accu']  
        
        token_data[sample_id] = {
            'token_ids': data['token_ids'],
            'law': law,   
            'accu': accu   
        }

print("处理样本数据...")
with open(samples_file_path, 'r', encoding='utf-8') as samples_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    
    first_sample_processed = False 

    for line in tqdm(samples_file, desc="处理样本", unit=" 行"):
        sample = json.loads(line)
        fact_id = sample['id']
        positive_id = sample['positive_sample']
        negative_samples = sample.get('negative_samples', [])
        
        if negative_samples:
            hard_negative_id = negative_samples[0]
        else:
            print(f"警告：样本 {fact_id} 没有硬负样本，将跳过此样本。")
            continue
        
        fact_data = token_data.get(fact_id)
        positive_data = token_data.get(positive_id)
        hard_negative_data = token_data.get(hard_negative_id)
        
        if fact_data is None:
            print(f"警告：未找到 fact id {fact_id} 的数据。")
            continue
        if positive_data is None:
            print(f"警告：未找到 positive sample id {positive_id} 的数据。")
            continue
        if hard_negative_data is None:
            print(f"警告：未找到 hard negative id {hard_negative_id} 的数据。")
            continue
        
        fact_label = fact_data['law'] * 62 + fact_data['accu']
        positive_label = positive_data['law'] * 62 + positive_data['accu']
        hard_negative_label = hard_negative_data['law'] * 62 + hard_negative_data['accu']
        
        processed_sample = {
            'id': fact_id,
            'fact_token_ids': fact_data['token_ids'],
            'fact_label': fact_label,
            'positive_id': positive_id,
            'positive_token_ids': positive_data['token_ids'],
            'positive_label': positive_label,
            'hard_negative_id': hard_negative_id,
            'hard_negative_token_ids': hard_negative_data['token_ids'],
            'hard_negative_label': hard_negative_label
        }
        
        output_file.write(json.dumps(processed_sample, ensure_ascii=False) + '\n')
        
        if not first_sample_processed:
            with open(first_sample_output_path, 'w', encoding='utf-8') as first_sample_file:
                json.dump(processed_sample, first_sample_file, ensure_ascii=False, indent=4)
            first_sample_processed = True

print(f"处理后的数据已保存到 {output_file_path}")
print(f"第一个样本的处理结果已保存到 {first_sample_output_path}")

with open(first_sample_output_path, 'r', encoding='utf-8') as f:
    first_sample = json.load(f)
    print("第一个样本的处理结果：")
    print(json.dumps(first_sample, ensure_ascii=False, indent=4))
