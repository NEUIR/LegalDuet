import pickle as pk
import numpy as np
import json
import os
import thulac

# 配置参数
max_sentence_len = 100  # 每个句子的最大长度
max_sentence_num = 15  # 每个文档的最大句子数

# 初始化分词器
Cutter = thulac.thulac(seg_only=True)

# 加载现有的词汇表
with open('w2id_thulac.pkl', 'rb') as f:
    word2id = pk.load(f)

print(f"词汇表大小: {len(word2id)}")

# 分句函数
def split_sentences(fact, max_sentence_len, max_sentence_num):
    words = fact
    sentences = []
    sentence = []
    for word in words:
        if word == '。' or len(sentence) == max_sentence_len:
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(word)
    if sentence:
        sentences.append(sentence)
    if len(sentences) > max_sentence_num:
        sentences = sentences[:max_sentence_num]
    while len(sentences) < max_sentence_num:
        sentences.append([])
    return sentences

# 处理数据
def process_data(file_list, word2id, max_sentence_len, max_sentence_num):
    for file_name in file_list:
        fact_lists = []
        law_label_lists = []
        accu_label_lists = []
        term_lists = []
        num = 0
        with open(f'../data/{file_name}_cs_big.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line)
                fact = line['fact_cut'].strip().split(' ')
                sentences = split_sentences(fact, max_sentence_len, max_sentence_num)
                id_list = []
                for sentence in sentences:
                    id_sentence = []
                    for word in sentence:
                        if word in word2id:
                            id_sentence.append(int(word2id[word]))
                        else:
                            id_sentence.append(int(word2id['UNK']))
                    while len(id_sentence) < max_sentence_len:
                        id_sentence.append(int(word2id['BLANK']))
                    id_list.append(id_sentence)
                
                while len(id_list) < max_sentence_num:
                    id_list.append([int(word2id['BLANK'])] * max_sentence_len)
                
                id_numpy = np.array(id_list)
                fact_lists.append(id_numpy)
                law_label_lists.append(line['law'])
                accu_label_lists.append(line['accu'])
                term_lists.append(line['term'])
                num += 1
            f.close()
        data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists, 'accu_label_lists': accu_label_lists, 'term_lists': term_lists}
        pk.dump(data_dict, open(f'{file_name}_processed_LSTM_thulac_big.pkl', 'wb'))
        print(f'{file_name} dataset is processed over\n')

process_data(['train', 'valid', 'test'], word2id, max_sentence_len, max_sentence_num)
