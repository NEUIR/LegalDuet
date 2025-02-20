# import json
# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 嵌入文件路径
# embedding_files = {
#     'BERT': 'embedding_bert_baseline.jsonl',
#     'BERT+LegalDuet': 'embedding_bert_lcr_lgr.jsonl',
#     'SAILER': 'embedding_sailer_baseline.jsonl',
#     'SAILER+LegalDuet': 'embedding_sailer_lcr_lgr.jsonl'
# }

# # 加载嵌入和标签
# def load_embeddings(file_path):
#     embeddings = []
#     labels = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             embeddings.append(data['embedding'])
#             labels.append(data['accu'])
#     return np.array(embeddings), np.array(labels)

# # 指控标签映射
# charge_mapping = {
#     1: "Disturbance",
#     12: "Robbery",
#     55: "Fraud",
#     108: "Murder",
#     110: "Theft",
#     111: "Assault"
# }

# # 设置绘图
# plt.figure(figsize=(16, 12))
# sns.set(style="white")  # 去掉网格线

# # 遍历每个模型，加载嵌入并绘制t-SNE图
# for i, (model_name, file_path) in enumerate(embedding_files.items(), 1):
#     # 加载嵌入数据
#     embeddings, labels = load_embeddings(file_path)
    
#     # 将标签转换为具体指控
#     labels = np.vectorize(charge_mapping.get)(labels)

#     # 使用t-SNE降维
#     tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, random_state=42)
#     embeddings_2d = tsne.fit_transform(embeddings)
    
#     # 创建子图
#     plt.subplot(2, 2, i)
#     unique_labels = np.unique(labels)
    
#     # 使用不同颜色表示不同指控类别
#     for label in unique_labels:
#         idx = labels == label
#         plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
#                     label=f"{label}", 
#                     alpha=0.5, 
#                     s=20,
#                     edgecolors='none')  # 去掉黑色边框

#     plt.title(f"{model_name} Embeddings Visualization")
#     plt.xticks([])  # 去掉x轴刻度标签
#     plt.yticks([])  # 去掉y轴刻度标签
#     plt.legend(title="Charges", loc="upper right", fontsize=8, markerscale=0.7)  # 添加图例标题

# # 调整布局
# plt.tight_layout()

# # 保存图片
# output_image_path = 'embedding_visualization_21.png'
# plt.savefig(output_image_path, dpi=300)
# print(f"可视化结果已保存为 {output_image_path}")
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

embedding_files = {
    'SAILER': 'embedding_sailer_baseline_test_all.jsonl',
    # 'SAILER+lcr': 'embedding_sailer_lcr.jsonl',
    # 'SAILER+lgr': 'embedding_sailer_lgr.jsonl'
    'SAILER+LegalDuet': 'embedding_sailer_lcr_lgr_test_all.jsonl'
}

def load_embeddings(file_path):
    embeddings = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embedding'])
            labels.append(data['accu'])
    return np.array(embeddings), np.array(labels)

charge_mapping = {
    1: "Disturbance",
    12: "Robbery",
    55: "Fraud",
    108: "Murder",
    110: "Theft",
    111: "Assault"
}

sns.set(style="white")  # 去掉网格线

for model_name, file_path in embedding_files.items():
    embeddings, labels = load_embeddings(file_path)
    labels = np.vectorize(charge_mapping.get)(labels)
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        idx = labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                    label=f"{label}", 
                    alpha=0.5, 
                    s=20,
                    edgecolors='none')  # 去掉黑色边框

    # plt.title(f"{model_name} Embeddings Visualization")
    plt.xticks([]) 
    plt.yticks([])  
    plt.tight_layout(pad=0.1)  # 适当减少空白
    # plt.legend(title="Charges", loc="upper right", fontsize=8, markerscale=0.7)  # 添加图例标题

    individual_output_path = f"embedding_visualization_{model_name}_test_all.pdf"
    plt.savefig(individual_output_path, dpi=300)
    plt.close()  
    print(f"{model_name} 的可视化结果已保存为 {individual_output_path}")
