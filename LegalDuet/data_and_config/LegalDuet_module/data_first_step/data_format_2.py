import json

input_file = 'processed_data.json'
output_file = 'processed_data_jsonl.jsonl'

def convert_json_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                item = json.loads(line.strip())
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write('\n')
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误: {e}")

    print(f"数据已成功转换为 JSONL 格式，存储在 {output_file}")

convert_json_to_jsonl(input_file, output_file)
