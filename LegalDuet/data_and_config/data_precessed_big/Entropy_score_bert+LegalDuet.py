import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
import numpy as np
import pickle as pk
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
        self.bert = contrastive_bert_model.bert  
        self.linearF = nn.Linear(768, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 103 + 119 + 12)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :] 
        hF = self.linearF(cls_output)
        logits = self.classifier(hF)
        return logits 

try:
    valid_data = pk.load(open('valid_processed_sailer.pkl', 'rb'))
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

valid_dataset = LegalDataset(valid_data)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')
model = ContrastiveBERTModel(bert_model, projection_dim=256)

checkpoint = torch.load('checkpoint_epoch_2_batch_17500.pth.tar')
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
legal_model = LegalModel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
legal_model.to(device)

print(f"Using device: {device}")  

legal_model.eval()
entropies = []

with torch.no_grad():
    for batch in tqdm(valid_loader, desc="Calculating Entropy"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        logits = legal_model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)  
        
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)  # 避免log(0)
        entropies.extend(entropy.cpu().numpy())

np.save('entropies_bert+LegalDuet.npy', entropies)

plt.hist(entropies, bins=50)
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title('Entropy Distribution')
plt.savefig('entropy_distribution_bert+LegalDuet.png')
plt.show()
