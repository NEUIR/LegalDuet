import pickle as pk
import numpy as np
import json
from transformers import BertTokenizer

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

file_list = ['train', 'valid', 'test']

for i in range(len(file_list)):
    fact_lists = []
    law_label_lists = []
    accu_label_lists = []
    term_lists = []
    num = 0
    with open('../data/{}_cs_bert_big.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            line = json.loads(line)
            fact = line['fact_cut'].strip().split(' ')
            # 使用BERT分词器将词转换为ID
            fact_ids = tokenizer.convert_tokens_to_ids(fact)
            if len(fact_ids) > 512:
                fact_ids = fact_ids[:512]
            else:
                fact_ids += [0] * (512 - len(fact_ids))

            if len(fact_ids) <= 10:
                print(fact)
                print(idx)
                continue

            fact_numpy = np.array(fact_ids)

            fact_lists.append(fact_numpy)
            law_label_lists.append(line['law'])
            accu_label_lists.append(line['accu'])
            term_lists.append(line['term'])
            num += 1
        f.close()
    data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists, 'term_lists': term_lists}
    pk.dump(data_dict, open('{}_processed_bert_big.pkl'.format(file_list[i]), 'wb'))
    print(num)
    print('{}_dataset is processed over'.format(file_list[i])+'\n')
