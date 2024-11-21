from transformers import AutoTokenizer, AutoModel
import torch
import json
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("../../SAILER_zh") 
model = AutoModel.from_pretrained("../../SAILER_zh")

def vectorize_tokenized_text(tokenized_text, max_length=512):
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text.split())
    
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]  
    else:
        token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))  

    return token_ids

input_file_path = '../../../../outside_data/Law_Case.jsonl'  
output_file_path = 'Law_Case_with_tokens.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing", unit=" lines"):
        data = json.loads(line)
        tokenized_text = data['contents'] 
        token_ids = vectorize_tokenized_text(tokenized_text)
        
        data['token_ids'] = token_ids
        
        output_data = {
            'contents': data['contents'],
            'token_ids': data['token_ids'],
            'law': data['law'],
            'accu': data['accu'],
            'id': data['id']
        }
        
        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"向量化后的数据已保存到 {output_file_path}")
