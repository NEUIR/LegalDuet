def merge_crime_attributes(paper_file, new_file, output_file):
    # 读取 acc_paper.txt 文件内容，并存储为字典
    crime_attributes = {}
    with open(paper_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 3:
                crime_name = parts[2]
                attributes = parts[3:]
                crime_attributes[crime_name] = attributes

    # 读取 new_accu.txt 文件内容，匹配并添加属性
    merged_lines = []
    with open(new_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            crime_name = parts[0]
            if crime_name in crime_attributes:
                merged_line = '\t'.join(parts + crime_attributes[crime_name])
                merged_lines.append(merged_line)
            else:
                merged_lines.append('\t'.join(parts))

    # 将结果保存到 new_accu_few-shot.txt
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in merged_lines:
            f.write(line + '\n')

# 文件路径
paper_file = 'accu_paper.txt'
new_file = 'new_accu.txt'
output_file = 'new_accu_few-shot.txt'

# 合并罪行属性
merge_crime_attributes(paper_file, new_file, output_file)
