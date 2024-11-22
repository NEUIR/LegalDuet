# from transformers import AutoTokenizer, AutoModel
# import torch
# import json
# from tqdm import tqdm
# import os

# # 确保使用GPU 3
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# # 检查 GPU 是否可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 加载模型和分词器，并将模型加载到 GPU
# tokenizer = AutoTokenizer.from_pretrained("../../SAILER_zh")
# model = AutoModel.from_pretrained("../../SAILER_zh").to(device)

# def vectorize_tokenized_text(tokenized_text, max_length=512):
#     token_ids = tokenizer.convert_tokens_to_ids(tokenized_text.split())
    
#     if len(token_ids) > max_length:
#         token_ids = token_ids[:max_length]
#     else:
#         token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))

#     input_ids = torch.tensor([token_ids]).to(device)
#     attention_mask = torch.tensor([[1 if i < len(token_ids) else 0 for i in range(max_length)]]).to(device)

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
#     return outputs.last_hidden_state[:, 0, :].squeeze()

# # 读取 JSONL 文件并进行向量化，并将结果保存
# input_file_path = '../../../../outside_data/Law_Case.jsonl'
# output_file_path = 'vectorized_Law_Case.jsonl'

# with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
#     for line in tqdm(infile, desc="Processing", unit=" lines"):
#         data = json.loads(line)
#         tokenized_text = data['contents']
#         vector = vectorize_tokenized_text(tokenized_text).cpu().numpy()  # 将向量转回 CPU 上以便保存
        
#         output_data = {
#             'contents': tokenized_text,
#             'vector': vector.tolist(),
#             'law': data['law'],
#             'accu': data['accu'],
#             'id': data['id']
#         }
        
#         outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

# print(f"向量化后的数据已保存到 {output_file_path}")

#####################################################################################
# # 只对第一个样本处理：
# from transformers import AutoTokenizer, AutoModel
# import torch
# import json
# import os

# # 确保使用GPU 3
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# # 检查 GPU 是否可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 加载模型和分词器，并将模型加载到 GPU
# tokenizer = AutoTokenizer.from_pretrained("../../SAILER_zh")
# model = AutoModel.from_pretrained("../../SAILER_zh").to(device)

# def vectorize_tokenized_text(tokenized_text, max_length=512):
#     token_ids = tokenizer.convert_tokens_to_ids(tokenized_text.split())
    
#     if len(token_ids) > max_length:
#         token_ids = token_ids[:max_length]
#     else:
#         token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))

#     input_ids = torch.tensor([token_ids]).to(device)
#     attention_mask = torch.tensor([[1 if i < len(token_ids) else 0 for i in range(max_length)]]).to(device)

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
#     return outputs.last_hidden_state[:, 0, :].squeeze()

# # 读取 JSONL 文件并进行向量化（仅处理第一个样本）
# input_file_path = '../../../../outside_data/Law_Case.jsonl'
# output_file_path = 'vectorized_first_Law_Case.jsonl'

# with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
#     first_line = infile.readline()  # 读取第一个样本
#     data = json.loads(first_line)
#     tokenized_text = data['contents']
#     vector = vectorize_tokenized_text(tokenized_text).cpu().numpy()  # 将向量转回 CPU 上以便保存
    
#     output_data = {
#         'contents': tokenized_text,
#         'vector': vector.tolist(),
#         'law': data['law'],
#         'accu': data['accu'],
#         'id': data['id']
#     }
    
#     outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

# print(f"第一个样本的向量化数据已保存到 {output_file_path}")



#################################################################################################
# from transformers import AutoTokenizer, AutoModel
# import torch
# import json
# import os

# # 确保使用GPU 3
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# # 检查 GPU 是否可用
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 加载模型和分词器，并将模型加载到 GPU
# tokenizer = AutoTokenizer.from_pretrained("../../SAILER_zh")
# model = AutoModel.from_pretrained("../../SAILER_zh").to(device)

# def vectorize_tokenized_text(tokenized_text, max_length=512):
#     token_ids = tokenizer.convert_tokens_to_ids(tokenized_text.split())
    
#     if len(token_ids) > max_length:
#         token_ids = token_ids[:max_length]
#     else:
#         token_ids = token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))

#     input_ids = torch.tensor([token_ids]).to(device)
#     attention_mask = torch.tensor([[1 if i < len(token_ids) else 0 for i in range(max_length)]]).to(device)

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
#     return outputs.last_hidden_state[:, 0, :].squeeze()

# # 指定要处理的样本ID
# target_ids = {200000}  # 示例ID，替换为你需要处理的实际ID

# # 读取 JSONL 文件并进行向量化（仅处理指定ID的样本）
# input_file_path = '../../../../outside_data/Law_Case.jsonl'
# output_file_path = 'vectorized_specific_Law_Cases.jsonl'

# with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
#     for line in infile:
#         data = json.loads(line)
#         if data['id'] in target_ids:
#             tokenized_text = data['contents']
#             vector = vectorize_tokenized_text(tokenized_text).cpu().numpy()  # 将向量转回 CPU 上以便保存
            
#             output_data = {
#                 'contents': tokenized_text,
#                 'vector': vector.tolist(),
#                 'law': data['law'],
#                 'accu': data['accu'],
#                 'id': data['id']
#             }
            
#             outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

# print(f"指定ID的样本向量化数据已保存到 {output_file_path}")
