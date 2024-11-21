import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import json
from tqdm import tqdm
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
num_epochs = 5
learning_rate = 1e-5
tau = 0.05 
validation_frequency = 500 
projection_dim = 256

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class LegalDataset(Dataset):
    def __init__(self, data_file):
        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample

def collate_fn(batch):
    max_length = 512  

    fact_ids = []
    positive_ids_lcr = []
    positive_ids_lgr = []
    negative_ids_lcr = []
    negative_ids_lgr = []
    fact_labels = []
    positive_labels_lcr = []
    positive_labels_lgr = []
    negative_labels_lcr = []
    negative_labels_lgr = []
    fact_sample_ids = []
    positive_sample_ids_lcr = []
    negative_sample_ids_lcr = []

    for sample in batch:
        fact_sample_ids.append(sample['id'])
        positive_sample_ids_lcr.append(sample['positive_id_lcr'])
        negative_sample_ids_lcr.append(sample['hard_negative_id_lcr'])

        fact_ids.append(torch.tensor(sample['fact_token_ids'][:max_length], dtype=torch.long))
        positive_ids_lcr.append(torch.tensor(sample['positive_token_ids_lcr'][:max_length], dtype=torch.long))
        positive_ids_lgr.append(torch.tensor(sample['positive_token_ids_lgr'][:max_length], dtype=torch.long))
        negative_ids_lcr.append(torch.tensor(sample['hard_negative_token_ids_lcr'][:max_length], dtype=torch.long))
        negative_ids_lgr.append(torch.tensor(sample['hard_negative_token_ids_lgr'][:max_length], dtype=torch.long))
        fact_labels.append(sample['fact_label'])
        positive_labels_lcr.append(sample['positive_label_lcr'])
        positive_labels_lgr.append(sample['positive_label_lgr'])
        negative_labels_lcr.append(sample['hard_negative_label_lcr'])
        negative_labels_lgr.append(sample['hard_negative_label_lgr'])
    
    fact_ids = torch.nn.utils.rnn.pad_sequence(fact_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    positive_ids_lcr = torch.nn.utils.rnn.pad_sequence(positive_ids_lcr, batch_first=True, padding_value=tokenizer.pad_token_id)
    negative_ids_lcr = torch.nn.utils.rnn.pad_sequence(negative_ids_lcr, batch_first=True, padding_value=tokenizer.pad_token_id)
    positive_ids_lgr = torch.nn.utils.rnn.pad_sequence(positive_ids_lgr, batch_first=True, padding_value=tokenizer.pad_token_id)
    negative_ids_lgr = torch.nn.utils.rnn.pad_sequence(negative_ids_lgr, batch_first=True, padding_value=tokenizer.pad_token_id)

    fact_attention_mask = (fact_ids != tokenizer.pad_token_id).long()
    positive_attention_mask_lcr = (positive_ids_lcr != tokenizer.pad_token_id).long()
    negative_attention_mask_lcr = (negative_ids_lcr != tokenizer.pad_token_id).long()
    positive_attention_mask_lgr = (positive_ids_lgr != tokenizer.pad_token_id).long()
    negative_attention_mask_lgr = (negative_ids_lgr != tokenizer.pad_token_id).long()

    return {
        'fact_ids': fact_ids,
        'fact_attention_mask': fact_attention_mask,
        'positive_ids_lcr': positive_ids_lcr,
        'positive_attention_mask_lcr': positive_attention_mask_lcr,
        'negative_ids_lcr': negative_ids_lcr,
        'negative_attention_mask_lcr': negative_attention_mask_lcr,
        'positive_ids_lgr': positive_ids_lgr,
        'positive_attention_mask_lgr': positive_attention_mask_lgr,
        'negative_ids_lgr': negative_ids_lgr,
        'negative_attention_mask_lgr': negative_attention_mask_lgr,
        'fact_sample_ids': fact_sample_ids,
        'positive_sample_ids_lcr': positive_sample_ids_lcr,
        'negative_sample_ids_lcr': negative_sample_ids_lcr,
        'fact_labels': fact_labels,
        'positive_labels_lcr': positive_labels_lcr,
        'negative_labels_lcr': negative_labels_lcr,
        'positive_labels_lgr': positive_labels_lgr,
        'negative_labels_lgr': negative_labels_lgr
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
    
train_file = 'train.jsonl'
valid_file = 'validation.jsonl'
train_dataset = LegalDataset(train_file)
valid_dataset = LegalDataset(valid_file)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

bert_model = BertModel.from_pretrained('bert-base-chinese')
model = ContrastiveBERTModel(bert_model, projection_dim=projection_dim)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []
best_valid_loss = float('inf')

debug_saved = False

best_model_path = 'best_model.pth'

def save_checkpoint(model, optimizer, epoch, batch_counter, filename):
    checkpoint = {
        'epoch': epoch + 1,
        'batch_counter': batch_counter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss
    }
    torch.save(checkpoint, filename)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_counter = 0  

    train_loss_temp = 0.0
    train_batch_counter = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        fact_ids = batch['fact_ids'].to(device)
        fact_attention_mask = batch['fact_attention_mask'].to(device)
        positive_ids_lcr = batch['positive_ids_lcr'].to(device)
        positive_attention_mask_lcr = batch['positive_attention_mask_lcr'].to(device)
        negative_ids_lcr = batch['negative_ids_lcr'].to(device)
        negative_attention_mask_lcr = batch['negative_attention_mask_lcr'].to(device)
        positive_ids_lgr = batch['positive_ids_lgr'].to(device)
        positive_attention_mask_lgr = batch['positive_attention_mask_lgr'].to(device)
        negative_ids_lgr = batch['negative_ids_lgr'].to(device)
        negative_attention_mask_lgr = batch['negative_attention_mask_lgr'].to(device)

        fact_labels = torch.tensor(batch['fact_labels'], dtype=torch.long, device=device)
        positive_labels_lcr = torch.tensor(batch['positive_labels_lcr'], dtype=torch.long, device=device)
        negative_labels_lcr = torch.tensor(batch['negative_labels_lcr'], dtype=torch.long, device=device)
        positive_labels_lgr = torch.tensor(batch['positive_labels_lgr'], dtype=torch.long, device=device)
        negative_labels_lgr = torch.tensor(batch['negative_labels_lgr'], dtype=torch.long, device=device)

        fact_sample_ids = torch.tensor(batch['fact_sample_ids'], dtype=torch.long, device=device)
        positive_sample_ids_lcr = torch.tensor(batch['positive_sample_ids_lcr'], dtype=torch.long, device=device)
        negative_sample_ids_lcr = torch.tensor(batch['negative_sample_ids_lcr'], dtype=torch.long, device=device)


        # 获取事实、正样本、负样本的嵌入表示
        fact_embeddings = model(input_ids=fact_ids, attention_mask=fact_attention_mask) 
        positive_embeddings_lcr = model(input_ids=positive_ids_lcr, attention_mask=positive_attention_mask_lcr)  
        negative_embeddings_lcr = model(input_ids=negative_ids_lcr, attention_mask=negative_attention_mask_lcr)  
        positive_embeddings_lgr = model(input_ids=positive_ids_lgr, attention_mask=positive_attention_mask_lgr)  
        negative_embeddings_lgr = model(input_ids=negative_ids_lgr, attention_mask=negative_attention_mask_lgr)  

        batch_size_actual = fact_embeddings.size(0)
        
        # 构建共享负样本池
        negative_pool_embeddings_lcr = torch.cat([
            fact_embeddings,
            positive_embeddings_lcr,
            negative_embeddings_lcr
        ], dim=0)  # [3 * batch_size, hidden_dim]

        # 构建共享负样本池
        negative_pool_embeddings_lgr = torch.cat([
            positive_embeddings_lgr,
            negative_embeddings_lgr
        ], dim=0)  # [2 * batch_size, hidden_dim]

        # 合并所有标签
        all_labels_lcr = torch.cat([fact_labels, positive_labels_lcr, negative_labels_lcr])
        # 标签掩码
        label_mask_lcr = (all_labels_lcr.unsqueeze(0) != fact_labels.unsqueeze(1)).float()
        # 合并所有ID
        all_sample_ids_lcr = torch.cat([fact_sample_ids, positive_sample_ids_lcr, negative_sample_ids_lcr])

        negative_labels_only_lgr = torch.cat([positive_labels_lgr, negative_labels_lgr])  # [2 * batch_size_actual]
        label_mask_lgr = (negative_labels_only_lgr.unsqueeze(0) != fact_labels.unsqueeze(1)).float()  # [batch_size_actual, 2 * batch_size_actual]

        # 创建自己和自己以及正样本的掩码
        self_pos_mask_lcr = torch.eye(batch_size_actual, device=device)
        self_pos_mask_lcr = torch.cat([self_pos_mask_lcr, self_pos_mask_lcr, torch.zeros_like(self_pos_mask_lcr)], dim=1)

        # 创建 self_pos_mask
        self_pos_mask_lgr = torch.zeros((batch_size_actual, 2 * batch_size_actual), device=device)
        # 掩盖自身的 B_i，在 negative_pool_embeddings 中的位置为 [0, batch_size_actual - 1]
        batch_indices_lgr = torch.arange(batch_size_actual, device=device)
        self_pos_mask_lgr[batch_indices_lgr, batch_indices_lgr] = 1  # 掩盖自身的 B_i

        # 创建重复样本掩码
        # duplicate_mask_lcr = torch.zeros((len(all_sample_ids_lcr), len(all_sample_ids_lcr)), device=device)
        # for i in range(len(all_sample_ids_lcr)):
        #     duplicate_mask_lcr[i] = (all_sample_ids_lcr == all_sample_ids_lcr[i]).float()
        # duplicate_mask_lcr = duplicate_mask_lcr.cumsum(dim=0) > 1  # 标记第二次及以后出现的重复
        # duplicate_mask_lcr = duplicate_mask_lcr.float()

        # # 对每个 fact 样本应用重复掩码
        # final_duplicate_mask_lcr = torch.zeros((batch_size_actual, len(all_sample_ids_lcr)), device=device)
        # for i in range(batch_size_actual):
        #     final_duplicate_mask_lcr[i] = duplicate_mask_lcr[i]

        # 合并所有掩码
        # final_mask_lcr = label_mask_lcr * (1 - self_pos_mask_lcr) * (1 - final_duplicate_mask_lcr)
        final_mask_lcr = label_mask_lcr * (1 - self_pos_mask_lcr) 

        # duplicate_mask_neg_lgr = torch.zeros((len(negative_labels_only_lgr), len(negative_labels_only_lgr)), device=device)
        # for i in range(len(negative_labels_only_lgr)):
        #     duplicate_mask_neg_lgr[i] = (negative_labels_only_lgr == negative_labels_only_lgr[i]).float()

        # duplicate_mask_neg_lgr = duplicate_mask_neg_lgr.cumsum(dim=0) > 1  # 标记第二次及以后出现的重复
        # duplicate_mask_neg_lgr = duplicate_mask_neg_lgr.float()

        # # 构建 final_duplicate_mask，大小为 [batch_size_actual, 2 * batch_size_actual]
        # final_duplicate_mask_lgr = torch.zeros((batch_size_actual, len(negative_labels_only_lgr)), device=device)
        # for i in range(batch_size_actual):
        #     final_duplicate_mask_lgr[i] = duplicate_mask_neg_lgr[i]

        # 合并所有掩码
        # final_mask_lgr = label_mask_lgr * (1 - self_pos_mask_lgr) * (1 - final_duplicate_mask_lgr)
        final_mask_lgr = label_mask_lgr * (1 - self_pos_mask_lgr)

        # 归一化嵌入向量
        fact_embeddings = F.normalize(fact_embeddings, p=2, dim=1)  # [batch_size, hidden_dim]
        positive_embeddings_lcr = F.normalize(positive_embeddings_lcr, p=2, dim=1)  # 归一化
        negative_pool_embeddings_lcr = F.normalize(negative_pool_embeddings_lcr, p=2, dim=1)  # [num_negatives, hidden_dim]
        positive_embeddings_lgr = F.normalize(positive_embeddings_lgr, p=2, dim=1)  # 归一化
        negative_pool_embeddings_lgr = F.normalize(negative_pool_embeddings_lgr, p=2, dim=1)  # [num_negatives, hidden_dim]

        sim_pos_lgr = torch.sum(fact_embeddings * positive_embeddings_lgr, dim=1) / tau
        sim_neg_lgr = torch.matmul(fact_embeddings, negative_pool_embeddings_lgr.T) / tau   # [batch_size, 3 * batch_size]
        
        # 计算相似度
        sim_pos_lcr = torch.sum(fact_embeddings * positive_embeddings_lcr, dim=1) / tau
        sim_neg_lcr = torch.matmul(fact_embeddings, negative_pool_embeddings_lcr.T) / tau   # [batch_size, 3 * batch_size]
        
        # 应用掩码
        sim_neg_lcr = sim_neg_lcr.masked_fill(final_mask_lcr == 0, float('-inf'))
        # 应用掩码
        sim_neg_lgr = sim_neg_lgr.masked_fill(final_mask_lgr == 0, float('-inf'))

        # 调试信息：在第一个 batch 时，保存相似度和 exp(sim) 值
        if not debug_saved and batch_idx == 0:
            sim_pos_cpu_lcr = sim_pos_lcr.detach().cpu().numpy()
            sim_neg_cpu_lcr = sim_neg_lcr.detach().cpu().numpy()
            sim_pos_cpu_lgr = sim_pos_lgr.detach().cpu().numpy()
            sim_neg_cpu_lgr = sim_neg_lgr.detach().cpu().numpy()
            exp_sim_pos_cpu_lcr = torch.exp(sim_pos_lcr).detach().cpu().numpy()
            exp_sim_neg_cpu_lcr = torch.exp(sim_neg_lcr).detach().cpu().numpy()
            exp_sim_pos_cpu_lgr = torch.exp(sim_pos_lgr).detach().cpu().numpy()
            exp_sim_neg_cpu_lgr = torch.exp(sim_neg_lgr).detach().cpu().numpy()

            with open('debug_info.txt', 'w') as f:
                f.write("sim_pos_lcr:\n")
                f.write(np.array2string(sim_pos_cpu_lcr, separator=', ', threshold=np.inf))
                f.write("\n\nsim_neg_lcr:\n")
                f.write(np.array2string(sim_neg_cpu_lcr, separator=', ', threshold=np.inf))
                f.write("\n\nexp(sim_pos_lcr):\n")
                f.write(np.array2string(exp_sim_pos_cpu_lcr, separator=', ', threshold=np.inf))
                f.write("\n\nexp(sim_neg_lcr):\n")
                f.write(np.array2string(exp_sim_neg_cpu_lcr, separator=', ', threshold=np.inf))
                f.write("sim_pos_lgr:\n")
                f.write(np.array2string(sim_pos_cpu_lgr, separator=', ', threshold=np.inf))
                f.write("\n\nsim_neg_lgr:\n")
                f.write(np.array2string(sim_neg_cpu_lgr, separator=', ', threshold=np.inf))
                f.write("\n\nexp(sim_pos_lgr):\n")
                f.write(np.array2string(exp_sim_pos_cpu_lgr, separator=', ', threshold=np.inf))
                f.write("\n\nexp(sim_neg_lgr):\n")
                f.write(np.array2string(exp_sim_neg_cpu_lgr, separator=', ', threshold=np.inf))
            debug_saved = True  # 只保存一次

        # 使用交叉熵计算损失
        logits_lcr = torch.cat([sim_neg_lcr, sim_pos_lcr.unsqueeze(1)], dim=1)   # [batch_size, 3 * batch_size + 1]
        labels_lcr = torch.full((batch_size_actual,), logits_lcr.size(1) - 1, dtype=torch.long).to(device)  # 正样本的位置在最后一列
        loss_lcr = F.cross_entropy(logits_lcr, labels_lcr)

        logits_lgr = torch.cat([sim_neg_lgr, sim_pos_lgr.unsqueeze(1)], dim=1)   # [batch_size, 2 * batch_size + 1]
        labels_lgr = torch.full((batch_size_actual,), logits_lgr.size(1) - 1, dtype=torch.long).to(device)  # 正样本的位置在最后一列
        loss_lgr = F.cross_entropy(logits_lgr, labels_lgr)

        loss = loss_lcr + loss_lgr

        if torch.isnan(loss):
            print("NaN detected in loss")
            print("Loss_lcr:", loss_lcr)
            print("sim_pos_lcr:", sim_pos_lcr)
            print("sim_neg_lcr:", sim_neg_lcr)
            print("logits_lcr:", logits_lcr)
            print("labels_lcr:", labels_lcr)
            print("Loss_lgr:", loss_lgr)
            print("sim_pos_lgr:", sim_pos_lgr)
            print("sim_neg_lgr:", sim_neg_lgr)
            print("logits_lgr:", logits_lgr)
            print("labels_lgr:", labels_lgr)


        total_loss += loss.item()
        train_loss_temp += loss.item()
        train_batch_counter += 1
        batch_counter += 1

        # 反向传播和优化
        loss.backward()
        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 更新进度条
        progress_bar.set_postfix({'current_avg_loss': train_loss_temp / train_batch_counter})

        # 定期进行验证
        if batch_counter % validation_frequency == 0:
            print(f"\nPerforming validation after {batch_counter} batches...")
            model.eval()
            total_valid_loss_temp = 0
            with torch.no_grad():
                for val_batch in tqdm(valid_dataloader, desc="Validation (during epoch)", leave=False):
                    # 将数据移动到设备
                    fact_ids_val = val_batch['fact_ids'].to(device)
                    fact_attention_mask_val = val_batch['fact_attention_mask'].to(device)
                    positive_ids_val_lcr = val_batch['positive_ids_lcr'].to(device)
                    positive_attention_mask_val_lcr = val_batch['positive_attention_mask_lcr'].to(device)
                    negative_ids_val_lcr = val_batch['negative_ids_lcr'].to(device)
                    negative_attention_mask_val_lcr = val_batch['negative_attention_mask_lcr'].to(device)
                    positive_ids_val_lgr = val_batch['positive_ids_lgr'].to(device)
                    positive_attention_mask_val_lgr = val_batch['positive_attention_mask_lgr'].to(device)
                    negative_ids_val_lgr = val_batch['negative_ids_lgr'].to(device)
                    negative_attention_mask_val_lgr = val_batch['negative_attention_mask_lgr'].to(device)
                    # 获取标签
                    fact_labels_val = torch.tensor(val_batch['fact_labels'], dtype=torch.long, device=device)
                    positive_labels_val_lcr = torch.tensor(val_batch['positive_labels_lcr'], dtype=torch.long, device=device)
                    negative_labels_val_lcr = torch.tensor(val_batch['negative_labels_lcr'], dtype=torch.long, device=device)
                    positive_labels_val_lgr = torch.tensor(val_batch['positive_labels_lgr'], dtype=torch.long, device=device)
                    negative_labels_val_lgr = torch.tensor(val_batch['negative_labels_lgr'], dtype=torch.long, device=device)
                    # 获取ID
                    fact_sample_ids_val = torch.tensor(val_batch['fact_sample_ids'], dtype=torch.long, device=device)
                    positive_sample_ids_val_lcr = torch.tensor(val_batch['positive_sample_ids_lcr'], dtype=torch.long, device=device)
                    negative_sample_ids_val_lcr = torch.tensor(val_batch['negative_sample_ids_lcr'], dtype=torch.long, device=device)

                    # 获取嵌入表示
                    fact_embeddings_val = model(input_ids=fact_ids_val, attention_mask=fact_attention_mask_val)  # 已投影的嵌入
                    positive_embeddings_val_lcr = model(input_ids=positive_ids_val_lcr, attention_mask=positive_attention_mask_val_lcr)  # 已投影的嵌入
                    negative_embeddings_val_lcr = model(input_ids=negative_ids_val_lcr, attention_mask=negative_attention_mask_val_lcr)  # 已投影的嵌入
                    positive_embeddings_val_lgr = model(input_ids=positive_ids_val_lgr, attention_mask=positive_attention_mask_val_lgr)  # 已投影的嵌入
                    negative_embeddings_val_lgr = model(input_ids=negative_ids_val_lgr, attention_mask=negative_attention_mask_val_lgr)  # 已投影的嵌入
                    batch_size_valid = fact_embeddings_val.size(0)

                    # 构建负样本池，基于标签去除正样本
                    negative_pool_embeddings_val_lcr = torch.cat([
                        fact_embeddings_val,
                        positive_embeddings_val_lcr,
                        negative_embeddings_val_lcr
                    ], dim=0)  # [3 * batch_size, hidden_dim]

                    negative_pool_embeddings_val_lgr = torch.cat([
                        positive_embeddings_val_lgr,
                        negative_embeddings_val_lgr
                    ], dim=0)  # [2 * batch_size, hidden_dim]

                    all_labels_val_lcr = torch.cat([fact_labels_val, positive_labels_val_lcr, negative_labels_val_lcr])
                    all_sample_ids_val_lcr = torch.cat([fact_sample_ids_val, positive_sample_ids_val_lcr, negative_sample_ids_val_lcr])
                    label_mask_val_lcr = (all_labels_val_lcr.unsqueeze(0) != fact_labels_val.unsqueeze(1)).float()

                    negative_labels_only_val_lgr = torch.cat([positive_labels_val_lgr, negative_labels_val_lgr])  # [2 * batch_size_valid]
                    label_mask_val_lgr = (negative_labels_only_val_lgr.unsqueeze(0) != fact_labels_val.unsqueeze(1)).float()  # [batch_size_valid, 2 * batch_size_valid]

                    # 创建掩码来排除自身和正样本
                    self_pos_mask_val_lcr = torch.eye(batch_size_valid, device=device)
                    self_pos_mask_val_lcr = torch.cat([self_pos_mask_val_lcr, self_pos_mask_val_lcr, torch.zeros_like(self_pos_mask_val_lcr)], dim=1)
                    
                    self_pos_mask_val_lgr = torch.zeros((batch_size_valid, 2 * batch_size_valid), device=device)
                    batch_indices_val_lgr = torch.arange(batch_size_valid, device=device)
                    self_pos_mask_val_lgr[batch_indices_val_lgr, batch_indices_val_lgr] = 1  # 掩盖自身的 B_i

                    final_mask_val_lcr = label_mask_val_lcr * (1 - self_pos_mask_val_lcr) 
                    final_mask_val_lgr = label_mask_val_lgr * (1 - self_pos_mask_val_lgr)
                    # # 创建重复样本掩码
                    # duplicate_mask_val_lcr = torch.zeros((len(all_sample_ids_val_lcr), len(all_sample_ids_val_lcr)), device=device)
                    # for i in range(len(all_sample_ids_val_lcr)):
                    #     duplicate_mask_val_lcr[i] = (all_sample_ids_val_lcr == all_sample_ids_val_lcr[i]).float()
                    # duplicate_mask_val_lcr = duplicate_mask_val_lcr.cumsum(dim=0) > 1  # 标记第二次及以后出现的重复
                    # duplicate_mask_val_lcr = duplicate_mask_val_lcr.float()

                    # # 对每个 fact 样本应用重复掩码
                    # final_duplicate_mask_val_lcr = torch.zeros((batch_size_valid, len(all_sample_ids_val_lcr)), device=device)
                    # for i in range(batch_size_valid):
                    #     final_duplicate_mask_val_lcr[i] = duplicate_mask_val_lcr[i]

                    # 合并所有掩码
                    # final_mask_val_lcr = label_mask_val_lcr * (1 - self_pos_mask_val_lcr) * (1 - final_duplicate_mask_val_lcr)
                    

                    # duplicate_mask_neg_val_lgr = torch.zeros((len(negative_labels_only_val_lgr), len(negative_labels_only_val_lgr)), device=device)
                    # for i in range(len(negative_labels_only_val_lgr)):
                    #     duplicate_mask_neg_val_lgr[i] = (negative_labels_only_val_lgr == negative_labels_only_val_lgr[i]).float()
                    # duplicate_mask_neg_val_lgr = duplicate_mask_neg_val_lgr.cumsum(dim=0) > 1  # 标记第二次及以后出现的重复
                    # duplicate_mask_neg_val_lgr = duplicate_mask_neg_val_lgr.float()

                    # # 对每个 fact 样本应用重复掩码
                    # final_duplicate_mask_val_lgr = torch.zeros((batch_size_valid, len(negative_labels_only_val_lgr)), device=device)
                    # for i in range(batch_size_valid):
                    #     final_duplicate_mask_val_lgr[i] = duplicate_mask_neg_val_lgr[i]
                    # 合并所有掩码
                    # final_mask_val_lgr = label_mask_val_lgr * (1 - self_pos_mask_val_lgr) * (1 - final_duplicate_mask_val_lgr)
                    
                    fact_embeddings_val = F.normalize(fact_embeddings_val, p=2, dim=1)  # [batch_size, hidden_dim]
                    positive_embeddings_val_lcr = F.normalize(positive_embeddings_val_lcr, p=2, dim=1) 
                    negative_pool_embeddings_val_lcr = F.normalize(negative_pool_embeddings_val_lcr, p=2, dim=1)  # [num_negatives, hidden_dim]
                    positive_embeddings_val_lgr = F.normalize(positive_embeddings_val_lgr, p=2, dim=1)  
                    negative_pool_embeddings_val_lgr = F.normalize(negative_pool_embeddings_val_lgr, p=2, dim=1)  # [num_negatives, hidden_dim]

                    # 计算相似度
                    sim_pos_val_lcr = torch.sum(fact_embeddings_val * positive_embeddings_val_lcr, dim=1) / tau
                    sim_neg_val_lcr = torch.matmul(fact_embeddings_val, negative_pool_embeddings_val_lcr.T) / tau  # [batch_size, 3 * batch_size]

                    # 应用掩码
                    sim_neg_val_lcr = sim_neg_val_lcr.masked_fill(final_mask_val_lcr == 0, float('-inf'))

                    logits_val_lcr = torch.cat([sim_neg_val_lcr, sim_pos_val_lcr.unsqueeze(1)], dim=1)  # [batch_size, 3 * batch_size + 1]
                    labels_val_lcr = torch.full((batch_size_valid,), logits_val_lcr.size(1) - 1, dtype=torch.long).to(device)  # 正样本的位置在最后一列

                    loss_val_lcr = F.cross_entropy(logits_val_lcr, labels_val_lcr)

                    sim_pos_val_lgr = torch.sum(fact_embeddings_val * positive_embeddings_val_lgr, dim=1) / tau
                    sim_neg_val_lgr = torch.matmul(fact_embeddings_val, negative_pool_embeddings_val_lgr.T) / tau  # [batch_size, 3 * batch_size]

                    # 应用掩码
                    sim_neg_val_lgr = sim_neg_val_lgr.masked_fill(final_mask_val_lgr == 0, float('-inf'))

                    # 使用交叉熵计算损失
                    logits_val_lgr = torch.cat([sim_neg_val_lgr, sim_pos_val_lgr.unsqueeze(1)], dim=1)  # [batch_size, 3 * batch_size + 1]
                    labels_val_lgr = torch.full((batch_size_valid,), logits_val_lgr.size(1) - 1, dtype=torch.long).to(device)  # 正样本的位置在最后一列

                    loss_val_lgr = F.cross_entropy(logits_val_lgr, labels_val_lgr)
                    
                    
                    loss_val = loss_val_lcr + loss_val_lgr


                    if torch.isnan(loss_val):
                        print("NaN detected in validation loss")
                        print("Loss_lcr:", loss_val_lcr)
                        print("sim_pos_lcr:", sim_pos_val_lcr)
                        print("sim_neg_lcr:", sim_neg_val_lcr)
                        print("logits_lcr:", logits_val_lcr)
                        print("labels_lcr:", labels_val_lcr)
                        print("Loss_lgr:", loss_val_lgr)
                        print("sim_pos_lgr:", sim_pos_val_lgr)
                        print("sim_neg_lgr:", sim_neg_val_lgr)
                        print("logits_lgr:", logits_val_lgr)
                        print("labels_lgr:", labels_val_lgr)
                        continue  
                    
                    total_valid_loss_temp += loss_val.item()

            avg_valid_loss_temp = total_valid_loss_temp / len(valid_dataloader)
            print(f"Validation Loss after {batch_counter} batches: {avg_valid_loss_temp:.4f}")

            valid_losses.append(avg_valid_loss_temp)

            avg_train_loss_temp = train_loss_temp / train_batch_counter
            train_losses.append(avg_train_loss_temp)
            print(f"Average Training Loss for last {validation_frequency} batches: {avg_train_loss_temp:.4f}")

            train_loss_temp = 0.0
            train_batch_counter = 0

            # 保存当前验证后的模型
            checkpoint_filename = f"checkpoint_epoch_{epoch+1}_batch_{batch_counter}.pth.tar"
            save_checkpoint(model, optimizer, epoch, batch_counter, filename=checkpoint_filename)
            print(f"Model checkpoint saved: {checkpoint_filename}")

            # 更新最佳验证损失并保存最佳模型
            if avg_valid_loss_temp < best_valid_loss:
                best_valid_loss = avg_valid_loss_temp
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with validation loss: {best_valid_loss:.4f}")

            model.train()  # 切换回训练模式

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss')
    plt.xlabel('Validation Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_validation_loss.png')
    plt.show()

    print(f"The best model was found with a validation loss of {best_valid_loss:.4f}")
