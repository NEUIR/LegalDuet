import json

def load_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def generate_legal_ground(law_idx, accu_idx, law_lines, accu_lines, id2law_lines, id2accu_lines):
    """生成 LegalGround 文本"""
    law_article_number = id2law_lines[law_idx]
    law_article_detail = law_lines[law_idx]
    accu_name = id2accu_lines[accu_idx]
    accu_detail = accu_lines[accu_idx]
    
    legal_ground_text = (
        f"根据中华人民共和国刑法中[第{law_article_number}条]法律法规, "
        f"[{law_article_detail}]"
        f"上述证据确凿，符合罪名认定[{accu_detail}]"
        f"综上所述，遂即依法判决[{accu_name}]。"
    )
    
    return legal_ground_text

def generate_all_combinations(law_file, accu_file, id2law_file, id2accu_file, output_file):
    law_lines = load_lines(law_file)
    accu_lines = load_lines(accu_file)
    id2law_lines = load_lines(id2law_file)
    id2accu_lines = load_lines(id2accu_file)

    id_counter = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for law_idx in range(59):  # law 的值是 0-58
            for accu_idx in range(62):  # accu 的值是 0-61
                legal_ground_text = generate_legal_ground(
                    law_idx, accu_idx, law_lines, accu_lines, id2law_lines, id2accu_lines
                )
                formatted_output = {
                    "id": id_counter,
                    "contents": legal_ground_text
                }
                outfile.write(json.dumps(formatted_output, ensure_ascii=False) + '\n')
                id_counter += 1

law_file = 'law.txt'
accu_file = 'accu.txt'
id2law_file = 'id2law.txt'
id2accu_file = 'id2accu.txt'
output_file = 'legal_ground_index.jsonl'

generate_all_combinations(law_file, accu_file, id2law_file, id2accu_file, output_file)

print(f"所有可能的LegalGround文本已生成并保存到 {output_file}")