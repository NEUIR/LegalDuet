import pickle as pk
import numpy as np
import json
from string import punctuation
import os

add_punc = '，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc
max_length = 512

def punc_delete(fact_list):
    fact_filtered = []
    for word in fact_list:
        if word not in all_punc:
            fact_filtered.append(word)
    return fact_filtered

# 构建词汇表
def build_vocab(data_files, vocab_size=10000):
    word_count = {}
    for file in data_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                dic = json.loads(line)
                fact = dic['fact_cut'].strip().split(' ')
                fact = punc_delete(fact)
                for word in fact:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1

    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    sorted_words = sorted_words[:vocab_size]
    word2id = {word: idx+1 for idx, (word, _) in enumerate(sorted_words)}
    word2id['UNK'] = len(word2id) + 1
    word2id['BLANK'] = 0
    return word2id

data_files = ['../data/train_cs.json', '../data/valid_cs.json', '../data/test_cs.json']
word2id = build_vocab(data_files)
print(f"词汇表大小: {len(word2id)}")

with open('w2id_thulac.pkl', 'wb') as f:
    pk.dump(word2id, f)
    f.close()

# 数据处理
def process_data(file_list, word2id):
    for file_name in file_list:
        fact_lists = []
        law_label_lists = []
        accu_label_lists = []
        term_lists = []
        num = 0
        with open(f'../data/{file_name}_cs.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)
                fact = line['fact_cut'].strip().split(' ')
                fact = punc_delete(fact)
                id_list = []
                word_num = 0
                for j in range(min(len(fact), max_length)):
                    if fact[j] in word2id:
                        id_list.append(int(word2id[fact[j]]))
                        word_num += 1
                    else:
                        id_list.append(int(word2id['UNK']))
                while len(id_list) < 512:
                    id_list.append(int(word2id['BLANK']))

                if word_num <= 10:
                    continue

                id_numpy = np.array(id_list)
                fact_lists.append(id_numpy)
                law_label_lists.append(line['law'])
                accu_label_lists.append(line['accu'])
                term_lists.append(line['term'])
                num += 1
            f.close()
        data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists, 'term_lists': term_lists}
        pk.dump(data_dict, open(f'{file_name}_processed_thulac.pkl', 'wb'))
        print(f'{file_name} dataset is processed over\n')

process_data(['train', 'valid', 'test'], word2id)
