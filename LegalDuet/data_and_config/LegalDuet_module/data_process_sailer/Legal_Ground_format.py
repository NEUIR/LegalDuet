import json
from tqdm import tqdm

legal_ground_samples_file = 'Legal_Ground_samples_ids.jsonl'
predicted_samples_file = 'predicted_samples.jsonl'
output_file = 'processed_samples.jsonl'

predicted_samples = {}
with open(predicted_samples_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        data = json.loads(line)
        predicted_samples[data['sample_id']] = {
            'law_samples': data['law_samples'],
            'accu_samples': data['accu_samples']
        }

with open(legal_ground_samples_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing samples"):
        data = json.loads(line)
        case_id = data['id']
        true_law = data['true_law']
        true_accu = data['true_accu']

        # 获取 predicted_samples 中的预测结果
        if case_id in predicted_samples:
            predicted_laws = predicted_samples[case_id]['law_samples']
            predicted_accus = predicted_samples[case_id]['accu_samples']

            # 处理法条的负样本
            if true_law in predicted_laws:
                predicted_laws.remove(true_law) 
                negative_laws = predicted_laws[:3]  
            else:
                negative_laws = predicted_laws[:3]  

            # 处理罪名的负样本
            if true_accu in predicted_accus:
                predicted_accus.remove(true_accu) 
                negative_accus = predicted_accus[:3]  
            else:
                negative_accus = predicted_accus[:3]  

            result = {
                'id': case_id,
                'true_law': true_law,
                'true_accu': true_accu,
                'negative_laws': negative_laws,
                'negative_accus': negative_accus
            }

            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"处理后的样本已保存到 {output_file}")
