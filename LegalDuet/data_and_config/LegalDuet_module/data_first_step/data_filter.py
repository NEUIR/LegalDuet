import json
import os

input_dir = 'data'
output_dir = 'filtered_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_documents = 0
filtered_out_documents = 0

for filename in os.listdir(input_dir):
    if filename.endswith('.jsonl'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                total_documents += 1
                try:
                    doc = json.loads(line)
                    if 'fact_cut' in doc and doc['fact_cut'].strip():  # Check if 'contents' is not empty
                        json.dump(doc, outfile, ensure_ascii=False)
                        outfile.write('\n')
                    else:
                        filtered_out_documents += 1
                except json.JSONDecodeError:
                    print(f'Error decoding JSON: {line}')
                    filtered_out_documents += 1

print(f'Filtering completed. Total documents processed: {total_documents}')
print(f'Documents filtered out (empty or invalid): {filtered_out_documents}')
