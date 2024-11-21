# import json

# file_list = ['train', 'valid', 'test']

# for i in range(len(file_list)):
#     num = 0
#     with open('../data/{}_cs_sailer.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             num += 1
#     print(f"{file_list[i]} dataset has {num} entries.")

import json

file_list = ['train', 'valid', 'test']

for i in range(len(file_list)):
    num = 0
    with open('../data/{}_cs_sailer_big.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            num += 1
    print(f"{file_list[i]} dataset has {num} entries.")
