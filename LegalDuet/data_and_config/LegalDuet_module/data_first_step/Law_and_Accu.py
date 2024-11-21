import json

def convert_to_index_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for idx, line in enumerate(infile):
            formatted_data = {
                "id": idx,
                "contents": line.strip()  # 移除每行末尾的换行符和空白
            }
            outfile.write(json.dumps(formatted_data, ensure_ascii=False) + '\n')

def main():
    law_input = 'law.txt'
    law_output = 'law_index.jsonl'
    accu_input = 'accu.txt'
    accu_output = 'accu_index.jsonl'
    
    convert_to_index_format(law_input, law_output)
    print(f"Law index saved to {law_output}")
    
    convert_to_index_format(accu_input, accu_output)
    print(f"Accu index saved to {accu_output}")

if __name__ == "__main__":
    main()
