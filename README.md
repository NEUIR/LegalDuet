# ⚖️ LegalDuet  
Official repository for the paper "LegalDuet: Learning Effective Representations for Legal Judgment Prediction via a Dual-View Contrastive Learning".

<p align="center">
    <a href="https://arxiv.org/abs/xxxxxx">📜 Paper</a> •
    <a href="http://cail.cipsc.org.cn/task_summit.html?raceID=1&cail_tag=2018">📂 Data</a> •
    <a href="https://huggingface.co/Xubqpanda/LegalDuet">🤗 Model</a>
</p>

## 1. Introduction
This repository provides resources for our paper **LegalDuet**, which proposes a new method to enhance the accuracy of **Legal Judgment Prediction (LJP)**. Our model leverages a **dual-view legal reasoning mechanism** designed to emulate a judge's reasoning process when analyzing legal cases. This approach involves:
- **Law Case Clustering**: Utilizing past legal decisions to inform current judgments.
- **Legal Decision Matching**: Extracting specific legal rules and triggers to improve prediction quality.

### 1.1 Benchmark
We used the **CAIL** benchmark, based on the **CAIL2018** dataset, to comprehensively evaluate legal judgment prediction models. 

Key tasks include:
- **Law Article Prediction**: Determining the correct legal articles applicable to a given case.
- **Charge Prediction**: Predicting the correct charge based on the criminal facts.
- **Imprisonment Prediction**: Estimating the sentence length based on case specifics.

### 1.2 Methodology
**LegalDuet** employs two key reasoning modules:
1. **Law Case Clustering**: Uses past cases and decisions to inform new judgments, identifying subtle differences between similar cases to refine predictions.
2. **Legal Decision Matching**: Focuses on the specific legal articles and charges related to a case, enabling a more structured legal decision-making process.

The model is pre-trained using these dual mechanisms, creating a more tailored embedding space for legal tasks.

![LegalDuet Model](https://github.com/NEUIR/LegalDuet/blob/main/LegalDuet/data_and_config/draw/demo.pdf)

## 2. Installation

   ```
   conda create -n LegalDuet_env python==3.8
   conda activate LegalDuet_env
   ```

Check out and install requirements.
   ```
   git clone https://github.com/NEUIR/LegalDuet.git
   cd LegalDuet
   pip install -r requirements.txt
   ```

## 3. Fine-Tuning 

To quickly start using our model, you can download our pretrained model from Hugging Face:
<a href="https://huggingface.co/Xubqpanda/LegalDuet">🤗 Model</a>

Once downloaded, navigate to the `Fine-Tuning` directory to begin fine-tuning:
   ```
   cd Fine-Tuning
   ```
For detailed instructions on how to use the pretrained model, refer to `Fine-Tuning/README.md`

## 4. Reproducing LegalDuet

To reproduce the **LegalDuet** pretraining process, you will need the pretraining data.

The pretraining data `rest_data.json` can be downloaded from the following link:<a href="http://cail.cipsc.org.cn/task_summit.html?raceID=1&cail_tag=2018">📂 Pretraining Dataset</a>

Once downloaded, navigate to the `LegalDuet` directory to begin reproducing:
   ```
   cd LegalDuet
   ```
For detailed instructions on how to use the pretrained model, refer to `LegalDuet/README.md`

## 5. Result 

### 5.1 **Embedding Visualization**:

We conducted a comparative study of embedding spaces to evaluate the discriminative power of LegalDuet embeddings. Using t-SNE, we visualized the embedding spaces of BERT, BERT+LegalDuet, and other ablation models, with the final visualization of SAILER+LegalDuet shown in the bottom-right.

![LegalDuet Model](https://github.com/NEUIR/LegalDuet/blob/main/LegalDuet/data_and_config/draw/Embedding_Visualization/embedding_visualization_bert_ablation.png)

### 5.2 **Outperforms on CAIL**:

The Legal Judgment Prediction Performance on the CAIL-small Dataset. The best evaluation results are highlighted in bold, and the underlined scores indicate the second-best results across all models.

![LegalDuet Model](https://github.com/NEUIR/LegalDuet/blob/main/LegalDuet/data_and_config/draw/Cail_small_result.png)

The Legal Judgment Prediction Performance on the CAIL-big Dataset. The best evaluation results are highlighted in bold, and the underlined scores indicate the second-best results across all models.

![LegalDuet Model](https://github.com/NEUIR/LegalDuet/blob/main/LegalDuet/data_and_config/draw/Cail_big_result.png)

## 6. Citation

Please cite the paper and star the repo if you use LegalDuet and find it helpful.

Feel free to contact 20223953@stu.neu.edu.cn or open an issue if you have any questions.

```
@article{LegalDuet2024,
      title={LegalDuet: Learning Effective Representations for Legal Judgment Prediction via a Dual-View Contrastive Learning}, 
      author={Buqiang Xu, Zhenghao Liu, Sijia Yao, Xinze Li, Yu Gu, Ge Yu},
      year={2024},
}
```