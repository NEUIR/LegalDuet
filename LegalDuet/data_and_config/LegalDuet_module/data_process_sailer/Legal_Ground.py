import faiss
import numpy as np
import json
from tqdm import tqdm

accu_index = faiss.read_index("accu_index/index")  # 替换为实际的accu索引文件路径
law_index = faiss.read_index("law_index/index")    # 替换为实际的law索引文件路径

input_file_path = 'vectorized_Law_Case.jsonl'
output_file_path = 'Legal_Ground_samples_ids.jsonl'

results = []

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="Processing"):
        data = json.loads(line)
        query_vector = np.array(data['vector'], dtype=np.float32)  # 从 'vector' 字段加载向量
        
        # 获取当前样本的正样本accu和law
        true_accu = data['accu']
        true_law = data['law']

        # 在accu索引中进行检索
        D_accu, I_accu = accu_index.search(np.array([query_vector]), k=4)
        negative_accus = [int(I_accu[0][i]) for i in range(4) if int(I_accu[0][i]) != true_accu][:3]

        # 在law索引中进行检索
        D_law, I_law = law_index.search(np.array([query_vector]), k=4)
        negative_laws = [int(I_law[0][i]) for i in range(4) if int(I_law[0][i]) != true_law][:3]

        # 保存结果，仅保存正负样本的 id
        result_entry = {
            'id': data['id'],
            'true_accu': true_accu,
            'true_law': true_law,
            'negative_accus': negative_accus,
            'negative_laws': negative_laws
        }
        results.append(result_entry)

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for result in results:
        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"负样本搜索结果的ID已保存到 {output_file_path}")
