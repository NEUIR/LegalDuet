# 定义数字到中文的映射
def number_to_chinese(num):
    units = ['', '十', '百', '千']
    nums = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    chinese_num = ''
    str_num = str(num)
    
    for i in range(len(str_num)):
        n = int(str_num[i])
        if n != 0:
            chinese_num += nums[n] + units[len(str_num) - i - 1]
        elif i != len(str_num) - 1 and str_num[i + 1] != '0':
            chinese_num += nums[n]
    
    # 去掉"一十"中的"一"
    if chinese_num.startswith("一十"):
        chinese_num = chinese_num[1:]
    
    return chinese_num

# 读取输入数字并转换为中文
numbers = [
    336, 314, 351, 224, 128, 223, 341, 382, 238, 266, 313, 340, 288, 172, 209,
    243, 143, 261, 124, 343, 291, 235, 393, 274, 240, 246, 133, 177, 310, 312,
    233, 236, 264, 225, 234, 328, 417, 151, 135, 348, 217, 134, 237, 262, 150,
    114, 196, 303, 392, 226, 267, 272, 205, 372, 350, 275, 164, 338, 292, 397,
    125, 290, 176, 141, 279, 192, 307, 115, 344, 171, 193, 198, 363, 210, 270,
    144, 347, 280, 118, 239, 228, 305, 130, 276, 213, 186, 245, 232, 175, 391,
    345, 258, 253, 163, 140, 293, 194, 342, 271, 384, 153, 277, 214
]

# 转换数字为中文
chinese_numbers = [number_to_chinese(num) for num in numbers]

# 将结果写入.txt文件
with open('law_chinese.txt', 'w', encoding='utf-8') as f:
    for cn in chinese_numbers:
        f.write(cn + '\n')

print("转换完成，结果已保存到law_chinese.txt。")
