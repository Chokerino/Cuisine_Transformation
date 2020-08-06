
import json
with open("data_v2.json",encoding="utf8") as file:
    data2=json.load(file)
    new_ings=[]
    for i in data2:
      temp={"id":0,"ingredients":[],"categories":[]}
      temp["id"]=i["recipe_id"]
      for j in i["tags"]:
        if j[1]=="ingredient":
          temp["ingredients"].append(j[0])
          temp["categories"].append(j[2])
          print(j[0])
      new_ings.append(temp)
      print(i)
      break
    
#print(new_ings)
#with open('formated_data_v2.json', 'w') as fp:
#        json.dump(new_ings, fp)      