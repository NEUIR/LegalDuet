import json
import random
from tqdm import tqdm
from itertools import product

with open('Legal_Ground.jsonl', 'r', encoding='utf-8') as infile:
    legal_ground_data = {json.loads(line)['id']: json.loads(line)['contents'] for line in infile}


input_file_path = 'processed_samples.jsonl'
output_file_path = 'Processed_Legal_Ground_Data_with_Hard_Negative.jsonl'
tokens_file_path = 'Law_Case_with_tokens.jsonl'
first_sample_output_file = 'Processed_Legal_Ground_Data_with_Hard_Negative_first_sample.json'

results = []
batch_size = 10000  
batch_counter = 0 
first_sample_processed = False
token_data = {}

print("加载已分词的数据...")
with open(tokens_file_path, 'r', encoding='utf-8') as tokens_file:
    for line in tqdm(tokens_file, desc="加载 tokens", unit=" 行"):
        data = json.loads(line)
        sample_id = data['id']
        
        token_data[sample_id] = {
            'token_ids': data['token_ids'],
        }


with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'a', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing samples"):
        data = json.loads(line)
        case_id = data['id']
        true_law = data['true_law']
        true_accu = data['true_accu']
        negative_accus = data['negative_accus']
        negative_laws = data['negative_laws']

        true_index = true_law * 62 + true_accu
        positive_text = legal_ground_data.get(true_index, "未知LegalGround文本")

        negative_texts = []
        negative_labels = [] 

        for neg_law in negative_laws[:3]:
            neg_index = neg_law * 62 + true_accu
            negative_text = legal_ground_data.get(neg_index, "未知LegalGround文本")
            negative_texts.append(negative_text)
            negative_labels.append(neg_index)

        for neg_accu in negative_accus[:3]:
            neg_index = true_law * 62 + neg_accu
            negative_text = legal_ground_data.get(neg_index, "未知LegalGround文本")
            negative_texts.append(negative_text)
            negative_labels.append(neg_index)

        for neg_accu, neg_law in product(negative_accus[:3], negative_laws[:3]):
            neg_index = neg_law * 62 + neg_accu
            negative_text = legal_ground_data.get(neg_index, "未知LegalGround文本")
            negative_texts.append(negative_text)
            negative_labels.append(neg_index)

        if negative_texts:  
            selected_index = random.randint(0, len(negative_texts) - 1)
            selected_negative_text = negative_texts[selected_index]
            selected_negative_label = negative_labels[selected_index]
        else:
            selected_negative_text = "未知LegalGround文本"
            selected_negative_label = None

        fact_data = token_data.get(case_id, {})
        fact_token_ids = fact_data.get('token_ids', [])
     
        result_entry = {
            "id": case_id,
            'fact_token_ids': fact_token_ids,
            'fact_label': true_index,
            "positive_token_ids": positive_text,
            'positive_label': true_index,
            "hard_negative_token_ids": selected_negative_text,
            'hard_negative_label': selected_negative_label
        }

        if not first_sample_processed:
            with open(first_sample_output_file, 'w', encoding='utf-8') as first_sample_file:
                first_sample_file.write(json.dumps(result_entry, ensure_ascii=False, indent=4))
            first_sample_processed = True

        results.append(result_entry)
        batch_counter += 1

        if batch_counter >= batch_size:
            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            results = []  
            batch_counter = 0 

    if results:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"包含 LegalGround 文本的样本结果已保存到 {output_file_path}")
print(f"第一个样本的处理结果已保存到 {first_sample_output_file}")
