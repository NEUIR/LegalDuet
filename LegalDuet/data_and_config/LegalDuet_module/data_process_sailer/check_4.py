import json

first_sample_file = 'tokenized_Legal_Ground_first_sample.json'

with open(first_sample_file, 'r', encoding='utf-8') as f:
    first_sample_data = json.load(f)

print("第一个样本的ID:", first_sample_data['id'])
print("正样本的分词内容:")
print(first_sample_data['tokenized_Legal_Ground_positive_contents'])

print("\n负样本的分词内容:")
for idx, neg_content in enumerate(first_sample_data['tokenized_Legal_Ground_negative_contents'], start=1):
    print(f"负样本 {idx}: {neg_content}")
