import json

input_file = 'rest_cs_bert.json'
output_file = 'processed_data.json'

def process_data(input_file, output_file):
    processed_data = []
    law_accu_combinations = {}

    with open(input_file, 'r', encoding='utf-8') as infile:
        for idx, line in enumerate(infile):
            entry = json.loads(line.strip())
            doc_id = entry.get('id', f'{idx}')
            doc_text = ''.join(entry.get('fact_cut', []))
            law = entry.get('law', '')
            accu = entry.get('accu', '')
            term = entry.get('term', '')
            combination = (law, accu)
            if combination in law_accu_combinations:
                law_accu_combinations[combination] += 1
            else:
                law_accu_combinations[combination] = 1

    filtered_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile:
        for idx, line in enumerate(infile):
            entry = json.loads(line.strip())
            doc_id = entry.get('id', f'{idx}')
            doc_text = ''.join(entry.get('fact_cut', []))
            law = entry.get('law', '')
            accu = entry.get('accu', '')
            term = entry.get('term', '')
            combination = (law, accu)
            if law_accu_combinations[combination] > 1:
                processed_entry = {
                    "id": doc_id,
                    "fact_cut": doc_text,
                    "law": law,
                    "accu": accu,
                    'term': term
                }
                processed_data.append(processed_entry)
            else:
                filtered_count += 1

            if idx % 10000 == 0:  
                print(f'Processed {idx} entries...')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in processed_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"数据处理完成，存储在 {output_file}")
    print(f"筛选掉的样本数量: {filtered_count}")

process_data(input_file, output_file)
