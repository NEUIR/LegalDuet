import json
from tqdm import tqdm
from itertools import product

with open('../../../../outside_data/Legal_Ground.jsonl', 'r', encoding='utf-8') as infile:
    legal_ground_data = {json.loads(line)['id']: json.loads(line)['contents'] for line in infile}

input_file_path = 'processed_samples.jsonl'
output_file_path = 'Legal_Ground.jsonl'

first_sample_output_file = 'Legal_Ground_first_sample.json'

results = []
batch_size = 10000  
batch_counter = 0
first_sample_processed = False

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

        for neg_law in negative_laws[:3]:
            neg_index = neg_law * 62 + true_accu
            negative_text = legal_ground_data.get(neg_index, "未知LegalGround文本")
            negative_texts.append(negative_text)

        for neg_accu in negative_accus[:3]:
            neg_index = true_law * 62 + neg_accu
            negative_text = legal_ground_data.get(neg_index, "未知LegalGround文本")
            negative_texts.append(negative_text)

        for neg_accu, neg_law in product(negative_accus[:3], negative_laws[:3]):
            neg_index = neg_law * 62 + neg_accu
            negative_text = legal_ground_data.get(neg_index, "未知LegalGround文本")
            negative_texts.append(negative_text)

        result_entry = {
            "id": case_id,
            "Legal_Ground_positive_contents": positive_text,
            "Legal_Ground_negative_contents": negative_texts  # 保留最终的15个负样本文本
        }

        # 如果这是第一个样本，则保存到单独的文件中
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
