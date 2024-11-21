# import os

# input_dir = 'sub_datasets/'
# output_dir = 'sub_datasets/'

# # 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)

# # 遍历子数据集，批量创建 Faiss 索引
# for dataset_file in os.listdir(input_dir):
#     if dataset_file.endswith('_dataset.jsonl'):
#         combo = dataset_file.replace('_dataset.jsonl', '')
#         input_file_path = os.path.join(input_dir, dataset_file)
#         output_index_path = os.path.join(output_dir, f"{combo}_index")
        
#         print(f"Processing {combo}...")

#         # 使用 os.system 来调用命令行
#         os.system(f"python -m pyserini.index.faiss --input {input_file_path} --output {output_index_path}")

# print("所有子数据集的索引已创建并保存。")
