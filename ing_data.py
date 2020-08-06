import json
import io
import pandas as pd
import math
ing_data = pd.read_csv("ing_id.csv")

ing_dict={}
for n,i in ing_data.iterrows():
    temp=""
    if isinstance(i["Dietrx_Category"],str):
        temp=i["Dietrx_Category"]
    else:
        temp=i["new_name"]
    if temp not in ing_dict:
        ing_dict[temp]=[]
    ing_dict[temp].append(i["new_name"])

with open('ing_cat_list.json', 'w') as fp:
        json.dump(ing_dict, fp)