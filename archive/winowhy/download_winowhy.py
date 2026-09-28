import requests

url = "https://raw.githubusercontent.com/HKUST-KnowComp/WinoWhy/master/winowhy.json"

r = requests.get(url)
r.raise_for_status()

with open("winowhy.json", "wb") as f:
    f.write(r.content)

print("WinoWhy downloaded successfully!")