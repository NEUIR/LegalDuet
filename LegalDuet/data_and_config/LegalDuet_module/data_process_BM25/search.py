import faiss
import numpy as np
import json
from tqdm import tqdm

index = faiss.read_index("faiss_index/index")  

input_file_path = 'data/test.jsonl'
output_file_path = 'search_results.jsonl'

results = []

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="Searching"):
        data = json.loads(line)
        query_vector = np.array(data['vector'], dtype=np.float32) 
        
        D, I = index.search(np.array([query_vector]), k=5)

        result_entry = {
            'id': data['id'],
            'results': [
                {'docid': int(I[0][i]), 'score': float(D[0][i])} for i in range(5)
            ]
        }
        results.append(result_entry)

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for result in results:
        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"搜索结果已保存到 {output_file_path}")
