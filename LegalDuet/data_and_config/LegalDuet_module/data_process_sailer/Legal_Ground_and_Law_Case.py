import json
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../../SAILER_zh')

law_case_data_path = '../../../../outside_data/Law_Case.jsonl' 
tokenized_law_case_file = 'tokenized_Law_Case.jsonl'  
tokenized_legal_ground_file = 'tokenized_Legal_Ground.jsonl'  
output_file_path = 'merged_tokenized_data.jsonl' 

with open(law_case_data_path, 'r', encoding='utf-8') as infile:
    original_law_case_data = {json.loads(line)['id']: json.loads(line)['contents'] for line in infile}

def tokenize_and_pad(text, tokenizer, max_length=512):
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids += [0] * (max_length - len(token_ids))
    return token_ids

with open(tokenized_law_case_file, 'r', encoding='utf-8') as infile:
    tokenized_law_case_data = {json.loads(line)['id']: json.loads(line) for line in infile}

with open(tokenized_legal_ground_file, 'r', encoding='utf-8') as infile:
    tokenized_legal_ground_data = {json.loads(line)['id']: json.loads(line) for line in infile}

with open(output_file_path, 'a', encoding='utf-8') as outfile:
    for case_id, law_case_content in tqdm(original_law_case_data.items(), desc="Merging tokenized data"):
        
        tokenized_original_law_case = tokenize_and_pad(law_case_content, tokenizer)

        tokenized_law_case = tokenized_law_case_data.get(case_id, {})
        tokenized_legal_ground = tokenized_legal_ground_data.get(case_id, {})

        merged_entry = {
            "id": case_id,
            "tokenized_original_Law_Case_contents": tokenized_original_law_case,
            "tokenized_Law_Case_positive_contents": tokenized_law_case.get('tokenized_Law_Case_positive_contents', []),
            "tokenized_Law_Case_negative_contents": tokenized_law_case.get('tokenized_Law_Case_negative_contents', []),
            "tokenized_Legal_Ground_positive_contents": tokenized_legal_ground.get('tokenized_Legal_Ground_positive_contents', []),
            "tokenized_Legal_Ground_negative_contents": tokenized_legal_ground.get('tokenized_Legal_Ground_negative_contents', [])
        }

        outfile.write(json.dumps(merged_entry, ensure_ascii=False) + '\n')

print(f"合并后的样本已保存到 {output_file_path}")
