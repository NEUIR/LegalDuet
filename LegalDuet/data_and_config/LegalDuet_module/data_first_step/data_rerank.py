import json

def regenerate_ids(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile):
            try:
                data = json.loads(line)
                if 'id' in data:
                    del data['id']
                data['id'] = line_number
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number + 1}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error on line {line_number + 1}: {str(e)}")

input_filepath = 'filtered_data.jsonl'
output_filepath = 'Law_Case.jsonl'

regenerate_ids(input_filepath, output_filepath)

print(f"ID fields regenerated and saved to {output_filepath}")
