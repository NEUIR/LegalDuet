import numpy as np
import matplotlib.pyplot as plt

# 加载熵数据
entropy_model1 = np.load('entropies_SAILER_test.npy')
entropy_model3 = np.load('entropies_SAILER+LegalDuet_test.npy')

# 设定 x 轴范围 0-1
entropy_model1 = entropy_model1[entropy_model1 <= 1]
entropy_model3 = entropy_model3[entropy_model3 <= 1]

# 计算频率直方图（归一化，使其适用于折线图）
bins = np.linspace(0, 1, 50)
hist_model1, _ = np.histogram(entropy_model1, bins=bins)
hist_model3, _ = np.histogram(entropy_model3, bins=bins)

# 计算 bin 的中心位置
bin_centers = (bins[:-1] + bins[1:]) / 2

# 绘制折线图
plt.figure(figsize=(24, 11))
plt.plot(bin_centers, hist_model1, marker='o', linestyle='-', linewidth=4, markersize=10, label='SAILER', color='#9B5C63')
plt.plot(bin_centers, hist_model3, marker='s', linestyle='-', linewidth=4, markersize=10, label='LegalDuet', color='#A5CCC7')

# 设置图例
plt.legend(fontsize=60)

# 设置轴标签和刻度
plt.xlabel('Entropy', fontsize=72)
plt.ylabel('Frequency', fontsize=72)
plt.xticks(fontsize=72)
plt.yticks(fontsize=72)

# 保存并显示图形
plt.tight_layout()
plt.savefig('entropy_comparison_line_test.png', dpi=300)
plt.savefig('entropy_comparison_line_test.pdf', dpi=300)
plt.show()
