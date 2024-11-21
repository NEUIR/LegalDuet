import json
from tqdm import tqdm
import os

input_file_path = 'vectorized_Law_Case.jsonl'
output_dir = 'sub_datasets/'

os.makedirs(output_dir, exist_ok=True)

frequent_combinations = {
    "56_20": 85003,
    "55_18": 161852,
    "31_6": 21383,
    "10_49": 7422,
    "8_13": 74841,
    "54_47": 2980,
    "46_13": 154892,
    "23_41": 40477,
    "1_53": 13805,
    "34_26": 4332,
    "18_40": 2157,
    "9_2": 2589,
    "40_29": 4529,
    "4_21": 5897,
    "16_16": 2716,
    "0_57": 8097,
    "6_36": 23501,
    "53_24": 10101,
    "43_15": 5656,
    "15_37": 3083,
    "60_11": 2667,
    "37_3": 8186,
    "61_55": 3230,
    "41_5": 3287,
    "25_25": 3633,
    "28_39": 2341,
    "2_25": 7435,
    "24_50": 5165,
    "36_54": 2170
}

for combo in frequent_combinations.keys():
    accu, law = map(int, combo.split('_'))
    
    output_file_path = f"{output_dir}{combo}_dataset.jsonl"

    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc=f"Processing {combo}"):
            data = json.loads(line)
            if data['accu'] != accu or data['law'] != law:
                outfile.write(line)

print("所有高频组合的子数据集已创建并保存。")