import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn import metrics
import numpy as np
import pickle as pk
from tqdm import tqdm
import matplotlib.pyplot as plt

# 确保使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def load_latest_checkpoint(model, optimizer, path='.'):
    checkpoints = [f for f in os.listdir(path) if f.startswith('bert-crime_xs_finetuned_term_model_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return 0
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    model.load_state_dict(torch.load(os.path.join(path, latest_checkpoint)))
    epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
    print(f"Resuming from epoch {epoch}")
    return epoch

# 超参数设置
batch_size = 64  # 根据论文设置的批次大小
max_epoch = 16
learning_rate = 5e-6
task = ['law', 'accu', 'time']

# 加载预处理后的数据
try:
    train_data = pk.load(open('../data_processed/train_processed_bert-crime_xs.pkl', 'rb'))
    valid_data = pk.load(open('../data_processed/valid_processed_bert-crime_xs.pkl', 'rb'))
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 检查数据格式
print(f"Train data length: {len(train_data['fact_list'])}")
print(f"Valid data length: {len(valid_data['fact_list'])}")

# 创建数据集和数据加载器
train_dataset = LegalDataset(train_data)
valid_dataset = LegalDataset(valid_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('../xs')
bert_model = BertModel.from_pretrained('../xs')

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

# 优化器
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 训练和验证
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 从最新的检查点继续训练
start_epoch = load_latest_checkpoint(model, optimizer)

print(f"Using device: {device}")  # 打印设备信息

train_losses = []
valid_losses = []

# 打开文件保存损失值
with open('bert-crime_xs_finetuned_term_losses.txt', 'a') as f:
    for epoch in range(start_epoch, max_epoch):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{max_epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            law_labels = batch['law'].to(device)
            accu_labels = batch['accu'].to(device)
            term_labels = batch['term'].to(device)
            
            optimizer.zero_grad()
            law_logits, accu_logits, time_logits = model(input_ids, attention_mask)
            
            law_loss = criterion(law_logits, law_labels)
            accu_loss = criterion(accu_logits, accu_labels)
            term_loss = criterion(time_logits, term_labels)
            
            loss = term_loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")
        f.write(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}\n")

        model.eval()
        total_valid_loss = 0
        predic_law, predic_accu, predic_time = [], [], []
        y_law, y_accu, y_time = [], [], []
        correct_tags_law = 0
        correct_tags_accu = 0
        correct_tags_time = 0
        total_tags = 0.0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                law_labels = batch['law'].to(device)
                accu_labels = batch['accu'].to(device)
                term_labels = batch['term'].to(device)
                
                law_logits, accu_logits, time_logits = model(input_ids, attention_mask)
                
                law_loss = criterion(law_logits, law_labels)
                accu_loss = criterion(accu_logits, accu_labels)
                term_loss = criterion(time_logits, term_labels)
                
                loss = term_loss
                total_valid_loss += loss.item()

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

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_valid_loss:.4f}")
        f.write(f"Epoch {epoch+1}, Validation Loss: {avg_valid_loss:.4f}\n")

        prediction = [predic_law, predic_accu, predic_time]
        y = [y_law, y_accu, y_time]
        correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time]
        accuracy, metric = evaluation_multitask(y, prediction, 3, correct_tags, total_tags)
        
        print(f'Validation Results for Epoch {epoch + 1}')
        for i in range(3):
            print(f'Accuracy for {task[i]} prediction is: {accuracy[i]:.2f}%')
            print(f'Other metrics for {task[i]} prediction is: {metric[i]}')
        
        # 分类报告
        for idx, name in enumerate(task):
            print(f'* Classification Report for {name}:')
            report = metrics.classification_report(y[idx], prediction[idx])
            print(report)

        # 保存模型
        torch.save(model.state_dict(), f"bert-crime_xs_finetuned_term_model_epoch_{epoch+1}.pt")
        print(f"Model saved for epoch {epoch+1}")

plt.plot(range(1, max_epoch + 1), train_losses, label='Training Loss')
plt.plot(range(1, max_epoch + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('bert-crime_xs_finetuned_term_loss_curve.png')
plt.show()
