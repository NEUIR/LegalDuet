import json

def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            # 尝试加载整个文件为 JSON 数组
            data = json.load(file)
        except json.JSONDecodeError:
            # 如果失败，逐行读取
            print(f"Failed to load {file_path} as a whole JSON file. Trying to load line by line.")
            file.seek(0)
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {e}")
    return data

def compare_structure(data1, data2, path=""):
    if isinstance(data1, dict) and isinstance(data2, dict):
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        if keys1 != keys2:
            print(f"Different keys at {path}: {keys1.symmetric_difference(keys2)}")
        for key in keys1.intersection(keys2):
            compare_structure(data1[key], data2[key], path + f".{key}")
    elif isinstance(data1, list) and isinstance(data2, list):
        if len(data1) != len(data2):
            print(f"Different list lengths at {path}: {len(data1)} != {len(data2)}")
        else:
            for index, (item1, item2) in enumerate(zip(data1, data2)):
                compare_structure(item1, item2, path + f"[{index}]")
    else:
        if type(data1) != type(data2):
            print(f"Different types at {path}: {type(data1)} != {type(data2)}")
        elif data1 != data2:
            print(f"Different values at {path}: {data1} != {data2}")

def main():
    old_file = "train_cs.json"
    new_file = "train_cs_new.json"

    old_data = load_json(old_file)
    new_data = load_json(new_file)

    print(f"Number of records in old file: {len(old_data)}")
    print(f"Number of records in new file: {len(new_data)}")

    if len(old_data) != len(new_data):
        print(f"Different number of records: {len(old_data)} != {len(new_data)}")
        return
    
    for index, (record_old, record_new) in enumerate(zip(old_data, new_data)):
        compare_structure(record_old, record_new, path=f"[{index}]")

if __name__ == "__main__":
    main()
