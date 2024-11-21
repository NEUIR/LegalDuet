import pickle as pk
import numpy as np
import json

file_path = '../legal_basis_data/train_processed_thulac_Legal_basis.pkl'
num_samples_to_view = 5

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

with open(file_path, 'rb') as f:
    data_dict = pk.load(f)

fact_lists = data_dict['fact_list']
law_label_lists = data_dict['law_label_lists']
accu_label_lists = data_dict['accu_label_lists']
term_lists = data_dict['term_lists']

for i in range(num_samples_to_view):
    print(f"Sample {i + 1}:")
    print("Fact list (index matrix):")
    print(np.array(fact_lists[i]))
    print("Law labels:", law_label_lists[i])
    print("Accusation labels:", accu_label_lists[i])
    print("Term:", term_lists[i])
    print("\n")

print(f"Total samples in the dataset: {len(fact_lists)}")

with open('../data/train_cs.json', 'r', encoding= 'utf-8') as f:
    for idx, line in enumerate(f.readlines()):
        if idx < num_samples_to_view:
            line = json.loads(line)
            fact = line['fact_cut']
            print(f"Original Fact {idx + 1}:", fact)
            print("\n")

