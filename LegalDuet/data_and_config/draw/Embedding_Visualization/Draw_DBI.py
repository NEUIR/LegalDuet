import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

embedding_file_paths = [
    ('BERT', 'embedding_bert_baseline_test_all.jsonl'),
    ('BERT-XS', 'embedding_bert_xs_test_all.jsonl'),
    ('LegalDuet', 'embedding_sailer_lcr_lgr_test_all.jsonl'),
    ('SAILER','embedding_sailer_baseline_test_all.jsonl')
]

dbi_values = []  

for model_name, embedding_file_path in embedding_file_paths:
    print(f"\nProcessing {model_name} embeddings for DBI calculation...")

    facts_by_charge = {} 

    with open(embedding_file_path, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, desc="Loading embeddings"):
            data = json.loads(line)
            embedding = np.array(data['embedding'], dtype='float32')
            accu_label = data['accu']

            if accu_label not in facts_by_charge:
                facts_by_charge[accu_label] = []
            facts_by_charge[accu_label].append(embedding)

    charge_centers = {}
    intra_class_distances = {}
    for charge, embeddings in facts_by_charge.items():
        embeddings = np.array(embeddings).astype('float32')
        center = np.mean(embeddings, axis=0)
        charge_centers[charge] = center
        intra_class_distances[charge] = np.mean([np.linalg.norm(embedding - center) for embedding in embeddings])

    charges = list(charge_centers.keys())
    inter_class_distances = {}
    for i in range(len(charges)):
        for j in range(i + 1, len(charges)):
            dist = np.linalg.norm(charge_centers[charges[i]] - charge_centers[charges[j]])
            inter_class_distances[(charges[i], charges[j])] = dist

    dbi_values_per_model = []
    for i, charge_i in enumerate(charges):
        max_ratio = 0
        for j, charge_j in enumerate(charges):
            if i != j:
                S_i = intra_class_distances[charge_i]
                S_j = intra_class_distances[charge_j]
                M_ij = inter_class_distances[(charge_i, charge_j)] if (charge_i, charge_j) in inter_class_distances else inter_class_distances[(charge_j, charge_i)]
                ratio = (S_i + S_j) / M_ij
                max_ratio = max(max_ratio, ratio)
        dbi_values_per_model.append(max_ratio)

    dbi_index = np.mean(dbi_values_per_model)
    dbi_values.append(dbi_index)

    print(f"{model_name} DBI: {dbi_index}")
    
model_names = [name for name, _ in embedding_file_paths]
colors = ['#b3b3b3', '#6b8c7a', '#b07d62'] 

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, dbi_values, color=colors)
# plt.xlabel("Embedding Model")
plt.ylabel("Davies-Bouldin Index (DBI)")
# plt.title("DBI Comparison for Different Embedding Models")

plt.ylim(0, 10)
# 在每个柱形上方添加数值标签，并稍微上移
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{bar.get_height():.2f}',
             ha='center', va='bottom', color='black', fontsize=15)  # 字体大小调整为 12，并上移

plt.tight_layout()  
plt.savefig("DBI_comparison.png") 
plt.show()