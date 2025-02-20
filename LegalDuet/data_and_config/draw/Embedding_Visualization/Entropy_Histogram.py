import numpy as np
import matplotlib.pyplot as plt

entropy_model1 = np.load('entropies_bert.npy')
entropy_model2 = np.load('entropies_bert-xs.npy')
entropy_model3 = np.load('entropies_bert+LegalDuet.npy')
entropy_model4 = np.load('entropies_SAILER.npy')
plt.figure(figsize=(10, 10))

# 绘制直方图
plt.hist(entropy_model1, bins=50, alpha=0.5, label='BERT-Chinese', color='blue')
plt.hist(entropy_model2, bins=50, alpha=0.5, label='BERT-xs', color='grey')
plt.hist(entropy_model4, bins=50, alpha=0.5, label='SAILER', color='green')
plt.hist(entropy_model3, bins=50, alpha=0.5, label='LegalDuet', color='red')
# 调整图例字体大小
plt.legend(fontsize=30)  # 图例字体大小调整为14

# 设置轴标签字体大小
plt.ylabel('Frequency', fontsize=36)

# 设置刻度字体大小
plt.xticks(fontsize=36)  # X 轴刻度字体大小
plt.yticks(fontsize=36)  # Y 轴刻度字体大小

# 保存和显示图形
plt.tight_layout()
plt.savefig('entropy_comparison_big.pdf', dpi=300)  # 保存高分辨率图片
plt.show()
