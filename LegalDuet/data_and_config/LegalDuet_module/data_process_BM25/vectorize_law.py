from transformers import AutoTokenizer, AutoModel
import torch
import json
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("../../SAILER_zh")
model = AutoModel.from_pretrained("../../SAILER_zh").to(device)

def vectorize_tokenized_text(tokenized_text, max_length=512):
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text.split())
    
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))

    input_ids = torch.tensor([token_ids]).to(device)
    attention_mask = torch.tensor([[1 if i < len(token_ids) else 0 for i in range(max_length)]]).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    return outputs.last_hidden_state[:, 0, :].squeeze()

input_file_path = '../../../../outside_data/Law.jsonl'
output_file_path = 'vectorized_Law.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile, desc="Processing", unit=" lines"):
        data = json.loads(line)
        tokenized_text = data['contents']
        vector = vectorize_tokenized_text(tokenized_text).cpu().numpy() 
        
        output_data = {
            'contents': tokenized_text,
            'vector': vector.tolist(),
            'id': data['id']
        }
        
        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"向量化后的数据已保存到 {output_file_path}")