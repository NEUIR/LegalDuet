import torch
import json
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel
import os
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContrastiveBERTModel(nn.Module):
    def __init__(self, bert_model):
        super(ContrastiveBERTModel, self).__init__()
        self.bert = bert_model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        return cls_embedding  
    
class LegalModel(nn.Module):
    def __init__(self, contrastive_bert_model):
        super(LegalModel, self).__init__()
        self.bert = contrastive_bert_model.bert

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output 

tokenizer = BertTokenizer.from_pretrained('SAILER_zh')

weight_paths = {
    'sailer_baseline': None,  # Baseline不需要加载额外权重
    #'lcr': 'checkpoint_lcr_epoch_1_batch_18500.pth.tar',
    #'lgr': 'checkpoint_lgr_epoch_1_batch_18500.pth.tar',
    #'lcr_lgr': 'checkpoint_epoch_2_batch_17500.pth.tar'
}

# # 定义生成嵌入的函数
# def generate_embedding(text, model):
#     inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length').to(device)
#     # 移除 token_type_ids，如果存在的话
#     inputs.pop('token_type_ids', None)
#     with torch.no_grad():
#         outputs = model(**inputs)  # 获取模型的输出
#     return outputs.cpu().numpy()  # 直接返回模型输出

def generate_embedding(text, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 提取[CLS]向量
    return cls_embedding


input_file_path = 'filtered_Law_Case_test_all.jsonl'
output_file_paths = {
    'sailer_baseline': 'embedding_sailer_baseline_test_all.jsonl',
    #'lcr': 'embedding_lcr.jsonl',
    #'lgr': 'embedding_lgr.jsonl',
    #'lcr_lgr': 'embedding_lcr_lgr.jsonl'
}

for model_name, weight_path in weight_paths.items():
    if model_name == 'sailer_baseline':
        model = AutoModel.from_pretrained('SAILER_zh').to(device)
    else:
        model = LegalModel(ContrastiveBERTModel(AutoModel.from_pretrained('SAILER_zh'))).to(device)
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval() 
    
    output_file_path = output_file_paths[model_name]
    
    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc=f"Processing {model_name}", unit=" lines"):
            data = json.loads(line)
            fact_text = data['fact_cut']
            accu_label = data['accu']

            embedding = generate_embedding(fact_text, model)

            output_data = {
                'embedding': embedding.tolist(), 
                'accu': accu_label
            }
            outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

    print(f"完成 {model_name} 的嵌入生成，结果已保存到 {output_file_path}")
