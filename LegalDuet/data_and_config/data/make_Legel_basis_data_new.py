import json
import pickle as pk

def preprocess_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    processed_data = {
        "fact": [],
        "accu": [],
        "law": [],
        "term": []
    }
    
    for sample in data:
        processed_data["fact"].append(sample['fact_cut'])
        processed_data["accu"].append(sample['accu'])
        processed_data["law"].append(sample['law'])
        processed_data["term"].append(sample['term'])
    
    with open(output_file, 'wb') as f:
        pk.dump(processed_data, f)

# 使用示例
if __name__ == "__main__":
    input_file = 'valid_cs.json'
    output_file = '../legal_basis_data/valid_processed_thulac_Legal_basis_new.pkl'
    
    preprocess_data(input_file, output_file)
    print("Data preprocessing completed and saved to", output_file)