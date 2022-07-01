import pandas as pd

df = pd.read_json('./train.json')
a = []
#print(len(df))
for i in range(len(df)):
    a.append(df["image_dict"][i]["level_2"])
print(set(a))