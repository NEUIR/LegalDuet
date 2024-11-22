## 1. Data Preprocessing
Follow these steps to preprocess the raw dataset into the required format for LegalDuet pretraining.

    ```
    cd LegalDuet/data_and_config/data
    python tongji_3_bert_rest.py
    cd LegalDuet/data_and_config/LegalDuet_module/data_first_step
    python data_format.py
    python data_format_2.py
    python data_filter.py
    python rerank.py
    cd LegalDuet/data_and_config/data_processed
    python data_pickle_bert_rest.py
    ```

## 2. Contrastive Learning Dataset Construction
We construct two datasets for contrastive learning: LGR (Legal Ground Retrieval) and LCR (Legal Case Retrieval).
### 2.1 LGR data

#### 2.1.1 Training a downstream model.
    To build the LGR dataset, first fine-tune a BERT model on the Cail-Big dataset:
    Format the dataset in LADAN's format (refer to `Fine-Tuning/example.py` for details).
    The Cail-Big dataset can be downloaded from the following link.

    <a href="http://cail.cipsc.org.cn/task_summit.html?raceID=1&cail_tag=2018">📂 Cail-Big Dataset</a>

#### 2.1.2 Indexing Legal Ground data.
    Use the following script to predict Legal Grounds for `rest_data`.

    ```
    cd LegalDuet\data_and_config\bert-base-chinese_module
    python BERT_Legal_Ground_index.py
    ```

### 2.2 LCR data

#### 2.2.1 Vectorize the Dataset

    ```
    cd LegalDuet/data_and_config/LegalDuet_module/data_process_BM25
    python vectorize.py
    ```
#### 2.2.2 Build a Faiss Index

    ```
    cd LegalDuet/data_and_config/LegalDuet_module/data_process_sailer
    python -m pyserini.index.faiss \
    --input vectorized_Law_Case.jsonl \
    --output law_case_index
    ```
#### 2.2.3 Retrieve samples

    ```
    cd LegalDuet/data_and_config/LegalDuet_module/data_process_sailer
    python Law_Case_total_gpu.py
    python add_special_positive.py
    python Law_Case_total_cpu.py
    python Law_Case_fix_cpu.py
    python convert_format_final.py
    ```

## 3. Tokenization
Tokenize the datasets for both LGR and LCR data. The tokenizer from SAILER and bert-base-chinese can be used interchangeably.

### 3.1 Tokenize LCR Data

    ```
    cd LegalDuet\data_and_config\LegalDuet_module\data_process_sailer
    python Law_Case_tokenizer.py
    cd LegalDuet/LegalDuet/data_and_config/LegalDuet_module/in-batch/data_process/Legal_Ground
    python token_fetch.py
    ```
### 3.2 Tokenize LGR Data

    ```
    cd LegalDuet\data_and_config\LegalDuet_module\data_process_sailer
    python Legal_Ground.py
    python Legal_Ground_format.py
    cd LegalDuet\data_and_config\LegalDuet_module\in-batch\data_process\Law_Case
    python token_fetch.py
    ```
### 3.3 Merge and Prepare for Pretraining

    ```
    cd LegalDuet\data_and_config\LegalDuet_module\in-batch\data_process\total
    python merge.py
    python split.py
    ```

## 4. Pretraining
    Finally, start pretraining using the preprocessed dataset.

    ```
    cd LegalDuet/data_and_config/LegalDuet_module/in-batch/data_process/total
    python pretrain.py
    ```