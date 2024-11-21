import numpy as np
import matplotlib.pyplot as plt

entropy_model1 = np.load('entropies_bert.npy')
entropy_model2 = np.load('entropies_bert-xs.npy')
entropy_model3 = np.load('entropies_bert+LegalDuet.npy')

plt.figure(figsize=(10, 6))

plt.hist(entropy_model1, bins=50, alpha=0.5, label='BERT', color='blue')
plt.hist(entropy_model2, bins=50, alpha=0.5, label='BERT-Crime', color='green')
plt.hist(entropy_model3, bins=50, alpha=0.5, label='LegalDuet', color='red')

plt.legend()
# plt.xlabel('Entropy')
plt.ylabel('Frequency')
# plt.title('Entropy Distribution of Three Models')

plt.savefig('entropy_comparison.pdf')
plt.show()
