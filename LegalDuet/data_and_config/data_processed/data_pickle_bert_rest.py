import pickle as pk
import numpy as np
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

file_name = 'Law_Case'

# 初始化存储列表
fact_lists = []
law_label_lists = []
accu_label_lists = []
term_lists = []

num = 0
with open(f'../data/{file_name}.jsonl', 'r', encoding='utf-8') as f:
    idx = 0 
    for line in f.readlines():
        idx += 1
        line = json.loads(line)  
        fact = line['fact_cut'].strip().split(' ') 

        fact_ids = tokenizer.convert_tokens_to_ids(fact)
        if len(fact_ids) > 512:
            fact_ids = fact_ids[:512] 
        else:
            fact_ids += [0] * (512 - len(fact_ids))  

        if len(fact_ids) <= 10:
            print(f"Filtered fact: {fact}")
            print(f"Line number: {idx}")
            continue

        fact_numpy = np.array(fact_ids)

        fact_lists.append(fact_numpy)
        law_label_lists.append(line['law'])
        accu_label_lists.append(line['accu'])
        term_lists.append(line['term'])
        num += 1

# 保存为 .pkl 文件
data_dict = {
    'fact_list': fact_lists,
    'law_label_lists': law_label_lists,
    'accu_label_lists': accu_label_lists,
    'term_lists': term_lists
}
output_file = f'{file_name}_processed_bert.pkl'
pk.dump(data_dict, open(output_file, 'wb'))

# 打印处理信息
print(f"Number of valid samples: {num}")
print(f"Processed dataset saved to: {output_file}")
print(f"{file_name}_dataset processing is complete.\n")
