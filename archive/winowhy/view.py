import json

with open("winowhy.json") as f:
    data = json.load(f)

print(len(data))                       # 273
print(json.dumps(data[0], indent=2))   # pretty-print the first example