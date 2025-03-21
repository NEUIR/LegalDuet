## 1. Load `bert-base-chinese`
Start by loading the `bert-base-chinese` model and tokenizer from Hugging Face's `transformers` library
```python
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')
```
## 2. Add a Projection Head
We introduce a projection head on top of the [CLS] token to generate embeddings suitable for contrastive learning. You can define this as follows:
```python
import torch.nn as nn

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

model = ContrastiveBERTModel(bert_model, projection_dim=256)
```
## 3. Load `LegalDuet`
Replace the state_dict of bert-base-chinese with the pre-trained checkpoint from LegalDuet:
```python
checkpoint = torch.load('LegalDuet.tar')
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
```
## 4. Training model 
To fine-tune model, ensure your dataset format follows the structure used in **LADAN's dataset**. You can also define your dataset format based on this structure.For the full training pipeline, see `example.py`.