import json
import os

input_file = 'processed_data_jsonl.jsonl'
output_file = 'filtered_data.jsonl'

total_documents = 0
filtered_out_documents = 0

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} does not exist.")

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        total_documents += 1
        try:
            doc = json.loads(line)
            
            if 'fact_cut' in doc and doc['fact_cut'].strip():
                json.dump(doc, outfile, ensure_ascii=False)
                outfile.write('\n')
            else:
                filtered_out_documents += 1
        except json.JSONDecodeError:
            print(f'Error decoding JSON: {line.strip()}')
            filtered_out_documents += 1

print(f'Filtering completed. Total documents processed: {total_documents}')
print(f'Documents filtered out (empty or invalid): {filtered_out_documents}')
print(f'Filtered data saved to: {output_file}')
