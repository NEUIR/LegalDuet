import json
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../../SAILER_zh')

input_file_path = 'updated_query_samples_result.jsonl'  
law_case_data_path = '../../../../outside_data/Law_Case.jsonl'  
output_file_path = 'tokenized_Law_Case.jsonl'

first_sample_output_file = 'tokenized_Law_Case_first_sample.json'

with open(law_case_data_path, 'r', encoding='utf-8') as infile:
    law_case_data = {json.loads(line)['id']: json.loads(line)['contents'] for line in infile}

def tokenize_and_pad(text, tokenizer, max_length=512):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids += [0] * (max_length - len(token_ids))
    return token_ids

results = []
batch_size = 10000  
batch_counter = 0  
first_sample_processed = False

# 处理 query_samples_result_cpu.jsonl 文件
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'a', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing samples"):
        data = json.loads(line)
        case_id = data['query_id']
        positive_sample_id = data['positive_sample']['sample_id']
        negative_sample_ids = [neg['sample_id'] for neg in data['negative_samples']]

        # 查找正样本内容并进行分词
        if positive_sample_id in law_case_data:
            positive_text = law_case_data[positive_sample_id]
            tokenized_positive_text = tokenize_and_pad(positive_text, tokenizer)
        else:
            tokenized_positive_text = []

        # 查找负样本内容并进行分词
        tokenized_negative_texts = []
        for neg_id in negative_sample_ids:
            if neg_id in law_case_data:
                negative_text = law_case_data[neg_id]
                tokenized_negative_texts.append(tokenize_and_pad(negative_text, tokenizer))
            else:
                tokenized_negative_texts.append([])

        # 生成处理后的结果
        result_entry = {
            "id": case_id,
            "tokenized_Law_Case_positive_contents": tokenized_positive_text,  # 正样本分词结果
            "tokenized_Law_Case_negative_contents": tokenized_negative_texts  # 负样本分词结果
        }

        # 如果这是第一个样本，则保存到单独的文件中
        if not first_sample_processed:
            with open(first_sample_output_file, 'w', encoding='utf-8') as first_sample_file:
                first_sample_file.write(json.dumps(result_entry, ensure_ascii=False, indent=4))
            first_sample_processed = True

        # 将更新后的结果添加到列表
        results.append(result_entry)
        batch_counter += 1

        # 每处理10000个样本就保存一次
        if batch_counter >= batch_size:
            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            results = []  
            batch_counter = 0  

    # 处理最后一批未保存的结果
    if results:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"包含分词后的Law_Case文本的样本结果已保存到 {output_file_path}")
print(f"第一个样本的处理结果已保存到 {first_sample_output_file}")
