import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AutoModel
import numpy as np
import pickle as pk
from tqdm import tqdm
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LegalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.fact_list = data['fact_list']
        self.law_labels = data['law_label_lists']
        self.accu_labels = data['accu_label_lists']
        self.term_labels = data['term_lists']

    def __len__(self):
        return len(self.fact_list)

    def __getitem__(self, idx):
        token_ids = self.fact_list[idx]
        if len(token_ids) > 512:
            token_ids = token_ids[:512]
        elif len(token_ids) < 512:
            token_ids = token_ids + [0] * (512 - len(token_ids))
        attention_mask = [1] * len(token_ids)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'law': torch.tensor(self.law_labels[idx], dtype=torch.long),
            'accu': torch.tensor(self.accu_labels[idx], dtype=torch.long),
            'term': torch.tensor(self.term_labels[idx], dtype=torch.long),
            'id': idx  # 保存样本ID
        }

def load_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    return labels

original_law_labels = load_labels('original_law_labels.txt')  # 118个法条
new_law_labels = load_labels('new_law_labels.txt')            # 59个法条

original_accu_labels = load_labels('original_accu_labels.txt')  # 130个罪名
new_accu_labels = load_labels('new_accu_labels.txt')            # 62个罪名

# 创建标签名称到索引的映射
original_law_to_index = {label: idx for idx, label in enumerate(original_law_labels)}
new_law_to_index = {label: idx for idx, label in enumerate(new_law_labels)}

original_accu_to_index = {label: idx for idx, label in enumerate(original_accu_labels)}
new_accu_to_index = {label: idx for idx, label in enumerate(new_accu_labels)}

# 创建原始索引到新标签索引的映射
law_mapping = {original_law_to_index[label]: new_law_to_index[label] for label in original_law_labels if label in new_law_to_index}
accu_mapping = {original_accu_to_index[label]: new_accu_to_index[label] for label in original_accu_labels if label in new_accu_to_index}

# 加载测试数据
test_data = pk.load(open('../data_processed/Law_Case_processed_bert.pkl', 'rb'))

test_dataset = LegalDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('../SAILER_zh')
bert_model = AutoModel.from_pretrained('../SAILER_zh')

# 模型定义
class LegalModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(LegalModel, self).__init__()
        self.bert = bert_model
        self.linearF = torch.nn.Linear(768, 256)
        self.classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(256, 118 + 130 + 12)  # 修改输出层大小以适应任务
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]
        hF = self.linearF(cls_output)
        logits = self.classifier(hF)
        law_logits = logits[:, :118]
        accu_logits = logits[:, 118:248]
        time_logits = logits[:, 248:]
        return law_logits, accu_logits, time_logits

model = LegalModel(bert_model)

model.load_state_dict(torch.load("sailer_finetuned_big_model_epoch_9.pt"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

output_file_path = 'predicted_samples.jsonl'

with torch.no_grad():
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for batch in tqdm(test_loader, desc="Processing samples"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            law_labels = batch['law'].to(device)
            accu_labels = batch['accu'].to(device)
            sample_id = batch['id'].item()  # 获取样本ID

            law_logits, accu_logits, time_logits = model(input_ids, attention_mask)

            valid_law_preds, valid_accu_preds = [], []
            k = 4

            while (len(valid_law_preds) < 4 or len(valid_accu_preds) < 4) and k <= 20:  # 找到4个样本后停止
                # 获取Top-K结果
                law_topk_preds = torch.topk(law_logits, k, dim=1)[1].cpu().numpy().flatten()
                accu_topk_preds = torch.topk(accu_logits, k, dim=1)[1].cpu().numpy().flatten()

                for idx in law_topk_preds:
                    if idx in law_mapping:
                        mapped_idx = law_mapping[idx]
                        if mapped_idx not in valid_law_preds and len(valid_law_preds) < 4:
                            valid_law_preds.append(mapped_idx)

                for idx in accu_topk_preds:
                    if idx in accu_mapping:
                        mapped_idx = accu_mapping[idx]
                        if mapped_idx not in valid_accu_preds and len(valid_accu_preds) < 4:
                            valid_accu_preds.append(mapped_idx)

                k += 1  # 增加k，继续获取更多的预测结果

            # 构建结果
            result = {
                'sample_id': sample_id,
                'law_samples': valid_law_preds,
                'accu_samples': valid_accu_preds
            }

            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"所有样本的预测结果已保存到 {output_file_path}")