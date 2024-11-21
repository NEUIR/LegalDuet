import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle as pk
from tqdm import tqdm
from sklearn import metrics
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
        attention_mask = [1 if token_id != 0 else 0 for token_id in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'law': torch.tensor(self.law_labels[idx], dtype=torch.long),
            'accu': torch.tensor(self.accu_labels[idx], dtype=torch.long),
            'term': torch.tensor(self.term_labels[idx], dtype=torch.long)
        }

def evaluation_multitask(y, prediction, task_num, correct_tags, total_tags):
    accuracy_ = []
    metrics_acc = []
    for x in range(task_num):
        accuracy_1 = correct_tags[x] / total_tags * 100
        accuracy_metric = metrics.accuracy_score(y[x], prediction[x])
        macro_recall = metrics.recall_score(y[x], prediction[x], average='macro')
        micro_recall = metrics.recall_score(y[x], prediction[x], average='micro')
        macro_precision = metrics.precision_score(y[x], prediction[x], average='macro')
        micro_precision = metrics.precision_score(y[x], prediction[x], average='micro')
        macro_f1 = metrics.f1_score(y[x], prediction[x], average='macro')
        micro_f1 = metrics.f1_score(y[x], prediction[x], average='micro')
        accuracy_.append(accuracy_1)
        metrics_acc.append(
            (accuracy_metric, macro_recall, micro_recall, macro_precision, micro_precision, macro_f1, micro_f1))
    return accuracy_, metrics_acc

task = ['law', 'accu', 'term']

# 加载测试数据
test_data = pk.load(open('../data_processed/test_processed_bert-crime_ms.pkl', 'rb'))

# 确认数据结构并打印部分数据检查数据集内容
print("Loaded test data sample:")
if isinstance(test_data, dict):
    keys = list(test_data.keys())[:5]
    for key in keys:
        print(f"{key}: {test_data[key][:5]}")  # 打印每个关键字段的前5个元素
else:
    print("Unknown data structure")

# 检查数据集长度
print(f"Test data length: {len(test_data['fact_list'])}")

# 创建数据集和数据加载器
test_dataset = LegalDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('../ms')
bert_model = BertModel.from_pretrained('../ms')

# 模型定义
class LegalModel(torch.nn.Module):
    def __init__(self, bert_model):
        super(LegalModel, self).__init__()
        self.bert = bert_model
        self.linearF = torch.nn.Linear(768, 256)
        self.classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(256, 103 + 119 + 12)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]
        hF = self.linearF(cls_output)
        logits = self.classifier(hF)
        law_logits = logits[:, :103]
        accu_logits = logits[:, 103:222]
        time_logits = logits[:, 222:]
        return law_logits, accu_logits, time_logits

model = LegalModel(bert_model)

# 加载训练好的模型参数
model.load_state_dict(torch.load("bert-crime_ms_finetuned_term_model_epoch_7.pt"))

# 模型评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

predic_law, predic_accu, predic_time = [], [], []
y_law, y_accu, y_time = [], [], []
correct_tags_law = 0
correct_tags_accu = 0
correct_tags_time = 0
total_tags = 0.0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        law_labels = batch['law'].to(device)
        accu_labels = batch['accu'].to(device)
        time_labels = batch['term'].to(device)

        law_logits, accu_logits, time_logits = model(input_ids, attention_mask)

        law_preds = torch.argmax(law_logits, dim=1)
        accu_preds = torch.argmax(accu_logits, dim=1)
        time_preds = torch.argmax(time_logits, dim=1)

        correct_tags_law += (law_preds == law_labels).sum().item()
        correct_tags_accu += (accu_preds == accu_labels).sum().item()
        correct_tags_time += (time_preds == time_labels).sum().item()
        total_tags += law_labels.size(0)

        predic_law.extend(law_preds.cpu().numpy())
        predic_accu.extend(accu_preds.cpu().numpy())
        predic_time.extend(time_preds.cpu().numpy())

        y_law.extend(law_labels.cpu().numpy())
        y_accu.extend(accu_labels.cpu().numpy())
        y_time.extend(time_labels.cpu().numpy())

prediction = [predic_law, predic_accu, predic_time]
y = [y_law, y_accu, y_time]
correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time]
accuracy, metric = evaluation_multitask(y, prediction, 3, correct_tags, total_tags)

print('Test Results')
for i in range(3):
    print(f'Accuracy for {task[i]} prediction is: {accuracy[i]:.2f}%')
    print(f'Other metrics for {task[i]} prediction is: {metric[i]}')

 # 分类报告
for idx, name in enumerate(task):
    print(f'* Classification Report for {name}:')
    report = metrics.classification_report(y[idx], prediction[idx])
    print(report)
    