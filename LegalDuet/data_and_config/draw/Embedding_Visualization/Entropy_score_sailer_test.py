import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AutoModel
import numpy as np
import pickle as pk
from tqdm import tqdm
from sklearn import metrics
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
try:
    test_data = pk.load(open('/data1/xubuqiang/data_processed_save/test_processed_sailer.pkl', 'rb'))
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

# 创建数据集和数据加载器
test_dataset = LegalDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('SAILER_zh')
bert_model = AutoModel.from_pretrained('SAILER_zh')

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
        cls_output = outputs[1]  # [CLS] token's output
        hF = self.linearF(cls_output)
        logits = self.classifier(hF)
        law_logits = logits[:, :103]
        accu_logits = logits[:, 103:222]
        time_logits = logits[:, 222:]
        return law_logits, accu_logits, time_logits

# 加载训练好的模型参数
model = LegalModel(bert_model)
model.load_state_dict(torch.load("sailer_finetuned_accu_model_epoch_5.pt"))

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

# 用于存储accu的熵值
accu_entropies = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        law_labels = batch['law'].to(device)
        accu_labels = batch['accu'].to(device)
        time_labels = batch['term'].to(device)

        # 前向传播
        law_logits, accu_logits, time_logits = model(input_ids, attention_mask)

        # 计算预测结果
        law_preds = torch.argmax(law_logits, dim=1)
        accu_preds = torch.argmax(accu_logits, dim=1)
        time_preds = torch.argmax(time_logits, dim=1)

        # 计算正确标签数量
        correct_tags_law += (law_preds == law_labels).sum().item()
        correct_tags_accu += (accu_preds == accu_labels).sum().item()
        correct_tags_time += (time_preds == time_labels).sum().item()
        total_tags += law_labels.size(0)

        # 计算accu的概率分布并计算熵
        accu_probs = torch.softmax(accu_logits, dim=-1)
        accu_entropy = -torch.sum(accu_probs * torch.log(accu_probs + 1e-10), dim=-1)  # 加上1e-10避免log(0)
        accu_entropies.extend(accu_entropy.cpu().numpy())

        # 收集预测结果和真实标签
        predic_law.extend(law_preds.cpu().numpy())
        predic_accu.extend(accu_preds.cpu().numpy())
        predic_time.extend(time_preds.cpu().numpy())

        y_law.extend(law_labels.cpu().numpy())
        y_accu.extend(accu_labels.cpu().numpy())
        y_time.extend(time_labels.cpu().numpy())

# 计算多任务分类准确率和其他指标
prediction = [predic_law, predic_accu, predic_time]
y = [y_law, y_accu, y_time]
correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time]
accuracy, metric = evaluation_multitask(y, prediction, 3, correct_tags, total_tags)

print('Test Results')
for i in range(3):
    print(f'Accuracy for {task[i]} prediction is: {accuracy[i]:.2f}%')
    print(f'Other metrics for {task[i]} prediction is: {metric[i]}')

# 输出分类报告
for idx, name in enumerate(task):
    print(f'* Classification Report for {name}:')
    report = metrics.classification_report(y[idx], prediction[idx])
    print(report)

np.save('entropies_SAILER_test.npy', accu_entropies)


# 绘制并保存accu熵分布
plt.hist(accu_entropies, bins=50)
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title('Entropy Distribution for Accu Prediction')
plt.savefig('entropy_distribution_SAILER_test.png')
plt.show()

