import json
import io
import pandas as pd
import ast
cuisine_info = pd.read_csv("data.csv")
print(cuisine_info[cuisine_info['Recipe_id']==2610].index.values.astype(int)[0])
cuisine_info=cuisine_info.drop(cuisine_info.columns[0],axis=1)

import jsonpickle
#print(len(cuisine_info[cuisine_info['Recipe_id']==3000].index.values.astype(int)),cuisine_info[cuisine_info['Recipe_id']==3000].index.values.astype(int))
list_id=[]
count=0
updated_data=[]
with open("clean_recipies.json",encoding="utf8") as file:
    data=json.load(file)
    for i in data:
        new_data={}
        x=cuisine_info[cuisine_info['Recipe_id']==int(i["recipe_id"])].index.values.astype(int)
        #print(x,int(i["recipe_id"]),i["recipe_id"])
        if(len(x)==0):
            count+=1
            list_id.append(int(i["recipe_id"]))
            continue
        #print(type(i["recipe_id"]),x)
        #print(cuisine_info.iloc[x]["Continent"],cuisine_info.iloc[x]["Region"],cuisine_info.iloc[x]["Sub Region"]cuisine_info.iloc[x]["Sub Region"])
        new_data["Ingredients"]=[]
        new_data["recipe_id"]=i["recipe_id"]
        new_data["Continent"]=cuisine_info.iloc[x[0]]["Continent"]
        new_data["Region"]=cuisine_info.iloc[x[0]]["Region"]
        new_data["Sub_Region"]=cuisine_info.iloc[x[0]]["Sub Region"]
        ingredi = ast.literal_eval(cuisine_info.iloc[x[0]]["items"])
        new_data["Ingredients"]=ingredi
        new_data["steps"]=i["steps"]
        updated_data.append(new_data)
    #print(data,type(data))
    #print(data)
    #data = jsonpickle.encode(data)
    #updated_data = [item for item in data if item['recipe_id'] not in list_id]
    print(count)
    with open('recipe_with_cuisine.json', 'w') as fp:
        json.dump(updated_data, fp)