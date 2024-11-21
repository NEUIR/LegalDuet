import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return " ".join(tokens)

def process_and_store_tokenized_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for idx, line in enumerate(infile):
            line = line.strip()  
            if not line: 
                continue
            
            tokenized_line = tokenize_text(line)
            data = {
                "id": idx,
                "contents": tokenized_line
            }
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

def main():
    law_input = 'law.txt'
    law_output = 'law_tokenized.jsonl'
    accu_input = 'accu.txt'
    accu_output = 'accu_tokenized.jsonl'
    
    process_and_store_tokenized_text(law_input, law_output)
    print(f"Tokenized law text saved to {law_output}")
    
    process_and_store_tokenized_text(accu_input, accu_output)
    print(f"Tokenized accu text saved to {accu_output}")

if __name__ == "__main__":
    main()
