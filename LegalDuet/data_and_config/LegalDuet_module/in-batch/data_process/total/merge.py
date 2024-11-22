import json
from tqdm import tqdm

lcr_file_path = "Processed_Law_Case_Data_with_Hard_Negative.jsonl"
lgr_file_path = "Processed_Legal_Ground_Data_with_Hard_Negative.jsonl"
output_file_path = "Merged_Law_Ground_Data.jsonl"
single_sample_file_path = "Single_Sample.json"

def load_jsonl(file_path):
    data = []
    print(f"Loading {file_path}...")
    with tqdm(total=sum(1 for _ in open(file_path, 'r', encoding='utf-8')), desc=f"Reading {file_path}") as pbar:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
                pbar.update(1)  
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def merge_datasets_by_id(lcr_data, lgr_data):
    merged_data = []

    for lcr_sample, lgr_sample in tqdm(zip(lcr_data, lgr_data), total=len(lcr_data), desc="Merging Datasets"):
        if lcr_sample['id'] == lgr_sample['id']:
          
            merged_sample = {
                "id": lcr_sample["id"],  
                "fact_token_ids": lcr_sample["fact_token_ids"],  
                "fact_label": lcr_sample["fact_label"], 

                "positive_token_ids_lgr": lgr_sample.get("positive_token_ids", []),
                "positive_label_lgr": lgr_sample.get("positive_label", None),
                "hard_negative_token_ids_lgr": lgr_sample.get("hard_negative_token_ids", []),
                "hard_negative_label_lgr": lgr_sample.get("hard_negative_label", None),

                "positive_id_lcr": lcr_sample.get("positive_id", None),
                "positive_token_ids_lcr": lcr_sample.get("positive_token_ids", []),
                "positive_label_lcr": lcr_sample.get("positive_label", None),
                "hard_negative_id_lcr": lcr_sample.get("hard_negative_id", None),
                "hard_negative_token_ids_lcr": lcr_sample.get("hard_negative_token_ids", []),
                "hard_negative_label_lcr": lcr_sample.get("hard_negative_label", None)
            }

            merged_data.append(merged_sample)
        else:
            print(f"ID mismatch: lcr id {lcr_sample['id']} and lgr id {lgr_sample['id']} do not match.")
            return None  

    return merged_data

print("Loading datasets...")
lcr_data = load_jsonl(lcr_file_path)
lgr_data = load_jsonl(lgr_file_path)
print("Datasets loaded.")

print("Merging datasets...")
merged_data = merge_datasets_by_id(lcr_data, lgr_data)

if merged_data:
    print("Saving merged dataset...")
    save_jsonl(merged_data, output_file_path)

    with open(single_sample_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data[0], f, ensure_ascii=False, indent=4)

    print(f"数据整合完成！整合后的数据集保存到 {output_file_path}，第一个样本保存到 {single_sample_file_path}。")
else:
    print("数据整合失败，请检查 ID 是否一致。")
