import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn import metrics
import pickle as pk
from tqdm import tqdm
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

# 定义模型架构（与训练时一致）
class LegalModel(nn.Module):
    def __init__(self, bert_model):
        super(LegalModel, self).__init__()
        self.bert = bert_model
        self.linearF = nn.Linear(768, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 118 + 130 + 12)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的输出
        hF = self.linearF(cls_output)
        logits = self.classifier(hF)
        law_logits = logits[:, :118]
        accu_logits = logits[:, 118:248]
        time_logits = logits[:, 248:]
        return law_logits, accu_logits, time_logits

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

try:
    test_data = pk.load(open('../../../../data_processed/test_processed_sailer_big.pkl', 'rb'))
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

batch_size = 64
test_dataset = LegalDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载 BERT 模型和自定义模型架构
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = BertModel.from_pretrained('bert-base-chinese')
legal_model = LegalModel(bert_model)
legal_model.to(device)

checkpoint_path = 'finetuned_big_model_epoch_8.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=device)
legal_model.load_state_dict(checkpoint['state_dict'])

criterion = torch.nn.CrossEntropyLoss()

legal_model.eval()
total_test_loss = 0
predic_law, predic_accu, predic_time = [], [], []
y_law, y_accu, y_time = [], [], []
correct_tags_law = 0
correct_tags_accu = 0
correct_tags_time = 0
total_tags = 0.0
task = ['law', 'accu', 'time']
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        law_labels = batch['law'].to(device)
        accu_labels = batch['accu'].to(device)
        term_labels = batch['term'].to(device)

        # 模型前向传播
        law_logits, accu_logits, time_logits = legal_model(input_ids, attention_mask)

        law_loss = criterion(law_logits, law_labels)
        accu_loss = criterion(accu_logits, accu_labels)
        time_loss = criterion(time_logits, term_labels)
        loss = law_loss + accu_loss + time_loss
        total_test_loss += loss.item()

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

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

prediction = [predic_law, predic_accu, predic_time]
y = [y_law, y_accu, y_time]
correct_tags = [correct_tags_law, correct_tags_accu, correct_tags_time]
accuracy, metric = evaluation_multitask(y, prediction, 3, correct_tags, total_tags)

print(f'Test Results:')
for i in range(3):
    print(f'Accuracy for {task[i]} prediction is: {accuracy[i]:.2f}%')
    print(f'Other metrics for {task[i]} prediction is: {metric[i]}')
