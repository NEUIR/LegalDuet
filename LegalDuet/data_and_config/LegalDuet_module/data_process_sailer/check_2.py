import json

file_path = '/home/xubuqiang/LegalDuet/outside_data/Law_Case.jsonl'

query_ids = {0, 419896, 17610}

results = []

with open(file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        sample = json.loads(line)
        if sample['id'] in query_ids:
            results.append(sample)

for result in results:
    print(f"Sample ID: {result['id']}")
    print(f"Law: {result.get('law')}")
    print(f"Accu: {result.get('accu')}")
    print(f"Contents: {result.get('contents')}\n")
