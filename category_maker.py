import json
import io
import pandas as pd
import ast

cate_ing={}

import glob
import os
path = 'text corpus/types/ingredient/'
for filename in glob.glob(os.path.join(path, '*.txt')):
    #print(filename)
    with open((filename), 'r') as file:
        temp=[]
        for i in file.readlines():
            temp.append(i[:len(i)-1])
        cate_ing[filename[29:-4]]=temp


path = 'text corpus/types/process/'
for filename in glob.glob(os.path.join(path, '*.txt')):
    #print(filename)
    with open((filename), 'r') as file:
        temp=[]
        for i in file.readlines():
            temp.append(i[:len(i)-1])
        cate_ing[filename[26:-4]]=temp

path = 'text corpus/types/utensil/'
for filename in glob.glob(os.path.join(path, '*.txt')):
    #print(filename)
    with open((filename), 'r') as file:
        temp=[]
        for i in file.readlines():
            temp.append(i[:len(i)-1])
        cate_ing[filename[26:-4]]=temp
print(cate_ing)
with open('categories_items.json', 'w') as fp:
        json.dump(cate_ing, fp)

