import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle as pk
from tqdm import tqdm
from sklearn import metrics

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义数据集类
class LegalDataset(Dataset):
    def __init__(self, data):
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

# 定义评估函数
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

# 加载测试数据
test_data = pk.load(open('../data_processed/test_processed_bert.pkl', 'rb'))

# 创建数据集和数据加载器
test_dataset = LegalDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载分词器和预训练模型
tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
bert_model = BertModel.from_pretrained('../bert-base-chinese')

# 定义模型
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

# 加载微调后的模型参数
model.load_state_dict(torch.load('bert_finetuned_accu_model_epoch_11.pt'))
model.to(device)
model.eval()

# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()

# 开始评估
task = ['law', 'accu', 'term']

predic_law, predic_accu, predic_time = [], [], []
y_law, y_accu, y_time = [], [], []
correct_tags_law = 0
correct_tags_accu = 0
correct_tags_time = 0
total_tags = 0.0

# 损失累积变量
total_test_loss = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        law_labels = batch['law'].to(device)
        accu_labels = batch['accu'].to(device)
        term_labels = batch['term'].to(device)

        # 前向传播
        law_logits, accu_logits, time_logits = model(input_ids, attention_mask)

        # 计算损失
        law_loss = criterion(law_logits, law_labels)
        accu_loss = criterion(accu_logits, accu_labels)
        time_loss = criterion(time_logits, term_labels)
        loss = accu_loss

        # 累加损失
        total_test_loss += loss.item()

        # 预测
        law_preds = torch.argmax(law_logits, dim=1)
        accu_preds = torch.argmax(accu_logits, dim=1)
        time_preds = torch.argmax(time_logits, dim=1)

        correct_tags_law += (law_preds == law_labels).sum().item()
        correct_tags_accu += (accu_preds == accu_labels).sum().item()
        correct_tags_time += (time_preds == term_labels).sum().item()
        total_tags += law_labels.size(0)

        predic_law.extend(law_preds.cpu().numpy())
        predic_accu.extend(accu_preds.cpu().numpy())
        predic_time.extend(time_preds.cpu().numpy())

        y_law.extend(law_labels.cpu().numpy())
        y_accu.extend(accu_labels.cpu().numpy())
        y_time.extend(term_labels.cpu().numpy())

# 计算评估指标
prediction = [predic_law, predic_accu, predic_time]
y = [y_law, y_accu, y_time]
correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time]
accuracy, metric = evaluation_multitask(y, prediction, 3, correct_tags, total_tags)

# 计算平均损失
avg_test_loss = total_test_loss / len(test_loader)

# 输出结果
print('Test Results')
print(f'Average Test Loss: {avg_test_loss:.4f}')
for i in range(3):
    print(f'Accuracy for {task[i]} prediction is: {accuracy[i]:.2f}%')
    print(f'Other metrics for {task[i]} prediction is: {metric[i]}')

# 输出分类报告
for idx, name in enumerate(task):
    print(f'* Classification Report for {name}:')
    report = metrics.classification_report(y[idx], prediction[idx])
    print(report)
