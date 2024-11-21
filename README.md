# ⚖️ LegalDuet  
Official repository for the paper "LegalDuet: Learning Effective Representations for Legal Judgment Prediction via a Dual-View Legal Reasoning".

<p align="center">
    <a href="https://arxiv.org/abs/xxxxxx">📜 Paper</a> •
    <a href="https://huggingface.co/datasets/LegalDuet-Dataset">🤗 Data</a> •
    <a href="https://huggingface.co/models/LegalDuet">🤖 Model</a>
</p>

## 1. Introduction
This repository provides resources for our paper **LegalDuet**, which proposes a new method to enhance the accuracy of **Legal Judgment Prediction (LJP)**. Our model leverages a **dual-view legal reasoning mechanism** designed to emulate a judge's reasoning process when analyzing legal cases. This approach involves:
- **Law Case Reasoning**: Utilizing past legal decisions to inform current judgments.
- **Legal Ground Reasoning**: Extracting specific legal rules and triggers to improve prediction quality.

### 1.1 Benchmark
We developed the **LegalDuet** benchmark, based on the **CAIL2018** dataset, to comprehensively evaluate legal judgment prediction models. Our approach improves upon previous methods by focusing on both legal context and specific case details.

Key tasks include:
- **Law Article Prediction**: Determining the correct legal articles applicable to a given case.
- **Charge Prediction**: Predicting the correct charge based on the criminal facts.
- **Imprisonment Prediction**: Estimating the sentence length based on case specifics.

![Dual-View Reasoning](https://github.com/yourusername/LegalDuet/blob/main/Figure/benchmark.png)

### 1.2 Methodology
**LegalDuet** employs two key reasoning modules:
1. **Law Case Reasoning**: Uses past cases and decisions to inform new judgments, identifying subtle differences between similar cases to refine predictions.
2. **Legal Ground Reasoning**: Focuses on the specific legal articles and charges related to a case, enabling a more structured legal decision-making process.

The model is pre-trained using these dual mechanisms, creating a more tailored embedding space for legal tasks.

![LegalDuet Model](https://github.com/Xubqpanda/LegalDuet/blob/main/LegalDuet/data_and_config/draw/demo.pdf)

## 2. Quick Start: Reproducing LegalDuet

To reproduce the **LegalDuet** pretraining process, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LegalDuet.git
   cd LegalDuet

2. **Navigate to the module directory**:
   ```bash
   cd LegalDuet/LegalDuet/data_and_config/LegalDuet_module/in-batch/data_process/total
   python pretrain.py
