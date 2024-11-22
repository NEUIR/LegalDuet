import json
from tqdm import tqdm

with open('../../../../outside_data/Law_Case.jsonl', 'r', encoding='utf-8') as infile:
    law_case_data = {json.loads(line)['id']: json.loads(line)['contents'] for line in infile}

input_file_path = 'Law_Case_samples_ids_final.jsonl'
output_file_path = 'Law_Case.jsonl'

first_sample_output_file = 'Law_Case_first_sample.json'

results = []
batch_size = 10000  # 每处理10000个样本就保存一次
batch_counter = 0  # 计数器
first_sample_processed = False

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'a', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing samples"):
        data = json.loads(line)
        case_id = data['id']
        positive_sample_id = data['positive_sample']
        negative_sample_ids = data['negative_samples']

        # 生成正样本文本
        positive_text = law_case_data.get(positive_sample_id, "未知Law Case文本")

        # 生成负样本文本
        negative_texts = []
        for neg_id in negative_sample_ids:
            negative_text = law_case_data.get(neg_id, "未知Law Case文本")
            negative_texts.append(negative_text)

        # 只保留id和正负样本文本
        result_entry = {
            "id": case_id,
            "Law_Case_positive_contents": positive_text,
            "Law_Case_negative_contents": negative_texts  # 保留最终的15个负样本文本
        }

        # 如果这是第一个样本，则保存到单独的文件中
        if not first_sample_processed:
            with open(first_sample_output_file, 'w', encoding='utf-8') as first_sample_file:
                first_sample_file.write(json.dumps(result_entry, ensure_ascii=False, indent=4))
            first_sample_processed = True

        results.append(result_entry)
        batch_counter += 1

        # 每处理10000个样本就保存一次
        if batch_counter >= batch_size:
            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            results = []  # 清空结果列表
            batch_counter = 0  # 重置计数器

    # 处理最后一批未保存的结果
    if results:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"包含 Law Case 文本的样本结果已保存到 {output_file_path}")
print(f"第一个样本的处理结果已保存到 {first_sample_output_file}")
