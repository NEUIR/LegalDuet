import json
import numpy as np
from tqdm import tqdm

embedding_file_path = 'embedding_bert_xs.jsonl'

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

dbi_values = []
for i, charge_i in enumerate(charges):
    max_ratio = 0
    for j, charge_j in enumerate(charges):
        if i != j:
            S_i = intra_class_distances[charge_i]
            S_j = intra_class_distances[charge_j]
            M_ij = inter_class_distances[(charge_i, charge_j)] if (charge_i, charge_j) in inter_class_distances else inter_class_distances[(charge_j, charge_i)]
            ratio = (S_i + S_j) / M_ij
            max_ratio = max(max_ratio, ratio)
    dbi_values.append(max_ratio)

dbi_index = np.mean(dbi_values)

print("Davies-Bouldin Index (DBI):", dbi_index)
