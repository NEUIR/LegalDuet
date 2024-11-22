import json
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../../SAILER_zh')

input_file_path = 'Legal_Ground.jsonl'
output_file_path = 'tokenized_Legal_Ground.jsonl'

first_sample_output_file = 'tokenized_Legal_Ground_first_sample.json'

results = []
batch_size = 10000  
batch_counter = 0  
first_sample_processed = False

def tokenize_and_pad(text, tokenizer, max_length=512):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids += [0] * (max_length - len(token_ids))
    return token_ids

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'a', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Tokenizing samples"):
        data = json.loads(line)
        case_id = data['id']
        
        # 对正样本和负样本的文本进行分词并处理长度
        positive_text = data['Legal_Ground_positive_contents']
        negative_texts = data['Legal_Ground_negative_contents']
        
        tokenized_positive_text = tokenize_and_pad(positive_text, tokenizer)
        tokenized_negative_texts = [tokenize_and_pad(neg_text, tokenizer) for neg_text in negative_texts]

        result_entry = {
            "id": case_id,
            "tokenized_Legal_Ground_positive_contents": tokenized_positive_text,  
            "tokenized_Legal_Ground_negative_contents": tokenized_negative_texts  
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

    # 处理最后一批未保存的结果
    if results:
        for result in results:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"包含分词后的LegalGround文本的样本结果已保存到 {output_file_path}")
print(f"第一个样本的处理结果已保存到 {first_sample_output_file}")
