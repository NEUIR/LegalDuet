import json

input_file = 'updated_query_samples_result.jsonl'

output_file = 'converted_query_samples_result.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            data = json.loads(line)

            converted_data = {
                "id": data["query_id"],
                "positive_sample": data["positive_sample"]["sample_id"],
                "negative_samples": [neg["sample_id"] for neg in data["negative_samples"]]
            }

            outfile.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {line.strip()}")
        except KeyError as e:
            print(f"Missing key in data: {e}")

print(f"转换完成，结果已保存到 {output_file}")
