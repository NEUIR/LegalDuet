## 1. Data preprocess 
    ```
    cd LegalDuet\data_and_config\data
    python tongji_3_bert_rest.py
    cd LegalDuet\data_and_config\LegalDuet_module\data_first_step
    python data_format.py
    python data_format_2.py
    python data_filter.py
    python rerank.py
    cd LegalDuet\data_and_config\data_processed
    python data_pickle_bert_rest.py
    ```
## 2. Contrastive learning dataset build

### 2.1 LGR data

#### 2.1.1 Training a downstream model.
    For building LGR contrastive learning dataset, you should train a downstream model at first.
    You can follow our `Fine-Tuning/example.py` to finetuning a Bert model on Cail-Big dataset, don't forget to format the dataset in LADAN's format.
    The Cail-Big dataset can be downloaded from the following link:
    <a href="http://cail.cipsc.org.cn/task_summit.html?raceID=1&cail_tag=2018">📂 Cail-Big Dataset</a>

#### 2.1.2 Indexing Legal Ground data.
    You can use our code to predict Legal Ground for rest_data.
    ```
    cd LegalDuet\data_and_config\bert-base-chinese_module
    python BERT_Legal_Ground_index.py
    ```

### 2.2 LCR data

#### 2.2.1 