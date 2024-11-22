import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from sklearn import metrics
import numpy as np
import pickle as pk
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def load_latest_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}.")
        return 0
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])  # 使用自定义模型加载 state_dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resuming from epoch {epoch}")
    return epoch

class ContrastiveBERTModel(nn.Module):
    def __init__(self, bert_model, projection_dim=256):
        super(ContrastiveBERTModel, self).__init__()
        self.bert = bert_model
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, projection_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        projected = self.projection(cls_embedding)
        return projected
    
class LegalModel(nn.Module):
    def __init__(self, contrastive_bert_model):
        super(LegalModel, self).__init__()
        self.bert = contrastive_bert_model.bert  # 使用带投影头模型中的BERT部分
        self.linearF = nn.Linear(768, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 103 + 119 + 12)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的输出
        hF = self.linearF(cls_output)
        logits = self.classifier(hF)
        law_logits = logits[:, :103]
        accu_logits = logits[:, 103:222]
        time_logits = logits[:, 222:]
        return law_logits, accu_logits, time_logits
    
batch_size = 64
max_epoch = 10
learning_rate = 5e-6
task = ['law', 'accu', 'time']

try:
    train_data = pk.load(open('../../../../data_processed/train_processed_sailer.pkl', 'rb'))
    valid_data = pk.load(open('../../../../data_processed/valid_processed_sailer.pkl', 'rb'))
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

train_dataset = LegalDataset(train_data)
valid_dataset = LegalDataset(valid_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')
model = ContrastiveBERTModel(bert_model, projection_dim=256)
checkpoint = torch.load('checkpoint_epoch_2_batch_17500.pth.tar')
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
legal_model = LegalModel(model)

optimizer = AdamW(legal_model.parameters(), lr=learning_rate)

criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
legal_model.to(device)

print(f"Using device: {device}")  # 打印设备信息

train_losses = []
valid_losses = []

with open('finetuned_law_losses.txt', 'a') as f:
    for epoch in range(0, max_epoch):
        legal_model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{max_epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            law_labels = batch['law'].to(device)
            accu_labels = batch['accu'].to(device)
            term_labels = batch['term'].to(device)
            
            optimizer.zero_grad()
            law_logits, accu_logits, time_logits = legal_model(input_ids, attention_mask)
            
            law_loss = criterion(law_logits, law_labels)
            accu_loss = criterion(accu_logits, accu_labels)
            time_loss = criterion(time_logits, term_labels)
            
            loss = law_loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")
        f.write(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}\n")

        legal_model.eval()
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
                
                law_logits, accu_logits, time_logits = legal_model(input_ids, attention_mask)
                
                law_loss = criterion(law_logits, law_labels)
                accu_loss = criterion(accu_logits, accu_labels)
                time_loss = criterion(time_logits, term_labels)
                
                loss = law_loss
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

        torch.save({'state_dict': legal_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}, f"finetuned_law_model_epoch_{epoch+1}.pth.tar")
        print(f"Model saved for epoch {epoch+1}")

plt.plot(range(1, max_epoch + 1), train_losses, label='Training Loss')
plt.plot(range(1, max_epoch + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('finetuned_law_loss_curve.png')
plt.show()
