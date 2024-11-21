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
    train_file = 'train_cs_big.json'
    test_file = 'test_cs_big.json'
    valid_file = 'valid_cs_big.json'

    output_train_file = '../legal_basis_data/train_processed_thulac_Legal_basis_new_big.pkl'
    output_test_file = '../legal_basis_data/test_processed_thulac_Legal_basis_new_big.pkl'
    output_valid_file = '../legal_basis_data/valid_processed_thulac_Legal_basis_new_big.pkl'
    
    preprocess_data(train_file, output_train_file)
    print("Data preprocessing completed and saved to", output_train_file)
    preprocess_data(test_file, output_test_file)
    print("Data preprocessing completed and saved to", output_test_file)
    preprocess_data(valid_file, output_valid_file)
    print("Data preprocessing completed and saved to", output_valid_file)