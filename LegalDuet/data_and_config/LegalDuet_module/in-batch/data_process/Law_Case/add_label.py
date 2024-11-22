import json
from tqdm import tqdm

tokens_file_path = 'Law_Case_with_tokens.jsonl'
samples_file_path = 'Law_Case_samples_ids_final.jsonl'
output_file_path = 'Law_Case_samples_ids_final_with_label.jsonl'

token_data = {}

print("加载已分词的数据...")
with open(tokens_file_path, 'r', encoding='utf-8') as tokens_file:
    for line in tqdm(tokens_file, desc="加载 tokens", unit=" 行"):
        data = json.loads(line)
        sample_id = data['id']
        token_data[sample_id] = {
            'law': data['law'],     
            'accu': data['accu']
        }

print("处理样本数据...")
with open(samples_file_path, 'r', encoding='utf-8') as samples_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    
    first_sample_processed = False 

    for line in tqdm(samples_file, desc="处理样本", unit=" 行"):
        sample = json.loads(line)
        fact_id = sample['id']
        positive_id = sample['positive_sample']
        
        fact_data = token_data.get(fact_id)
        positive_data = token_data.get(positive_id)
        
        if fact_data is None:
            print(f"警告：未找到 fact id {fact_id} 的数据。")
            continue
        if positive_data is None:
            print(f"警告：未找到 positive sample id {positive_id} 的数据。")
            continue
        
        processed_sample = {
            'id': fact_id,
            'fact_law': fact_data['law'],      
            'fact_accu': fact_data['accu'],
            'positive_id': positive_id,
            'positive_law': positive_data['law'],   
            'positive_accu': positive_data['accu']
        }
        
        output_file.write(json.dumps(processed_sample, ensure_ascii=False) + '\n')

print(f"处理后的数据已保存到 {output_file_path}")
