
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split, cross_validate,StratifiedKFold
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
#from sklearn.pipeline import Pipel
import pandas as pd
import numpy as np
import ast
import string
import nltk
import matplotlib.pyplot as plt
#nltk.download('wordnet')


"""Moving ingredients and Cuisines to lists"""


cuisines=[]
ing_list=[]
recipe_names=[]
with open("Data/recipe_with_cuisine2.json",encoding="utf8") as file:
    data=json.load(file)
    for i in data:
        words=[]
        #print(i["Ingredients"],type(i["Ingredients"]))
        for j in i["Ingredients"]:
            #print(j)
            words.append(j)
        ing_list.append(words)
        #cuisines.append(i["Continent"])              #for continent wise label
        cuisines.append(i["Region"])                  #region wise label
        recipe_names.append(i["recipe_name"])

"""Cuisines to lowercase & more"""

for n, i in enumerate(cuisines):
  if i=="UK":
    cuisines[n]="United Kingdom"
  elif i=="spanish and portuguese":
    cuisines[n]="spanish"
  i.replace(" ","")
  cuisines[n]=cuisines[n].lower()
#print(len(ing_list))
print(len(cuisines))

"""Remove cooking instr"""

cook_list=["container","prepare","all","thread","and","drop","ladle","with","heat","cook","remove","cool","blend","push","stir",'saute', 'evaporated',"melt","store","foil","press","dry","put","break","seal","top","mash","toast","reduce","crush","tablespoon","break","crush","beat","sift","knife","board","set","divide","combine","toss","seperate","fry","bowl","fork","cut","sheet","season","oven","scoop","slice","broil","pierce","wash","dressing","whisk","coat","refrigerate","coat","preheat","transfer","chill","roll","cup","spoon","saucepan","smooth","pour","stirring","add","boil","simmer","pot","fold","skillet","pan","spatula","cover","soak","drain","serve","mix","place",'knead','separate','spread','bake','sheet','sheet','process','drizzle','processor','chop','sprinkle','garnish','peel','taste','blender',"dish"]

ing_list_cleaned=[]
x=[]
for n, i in enumerate(ing_list):
  temp=[]
  
  for n2, j in enumerate(i):
    if j not in cook_list:
      temp.append(j)
  x.append(len(temp))
  ing_list_cleaned.append(temp)


recipe_names_corrected=[]
for i in recipe_names:
  temp=[]
  i=i.replace("(","")
  i=i.replace(")","")
  i=i.replace("-","")
  i=i.split(" ")
  recipe_names_corrected.append(i)

print(len(recipe_names_corrected),len(ing_list_cleaned))

#replace spaces with underscores
temp1=[]
for i in ing_list_cleaned:
  temp=[]
  for j in i:
    temp.append(j.replace(" ",""))
  temp1.append(temp)
ing_list_cleaned=temp1

"""modify for training"""

temp=[]
for n,i in enumerate(cuisines):
    if i=="middle eastern":
        temp.append(ing_list_cleaned[n])

from sklearn.model_selection import train_test_split
ing_list_train_me, ing_list_test_me = train_test_split(temp, test_size=0.3, random_state=42)
temp=[]
for i in ing_list_train_me:
    temp2=""
    for j in i:
        temp2+=(j+" ")
    temp.append(temp2)
ing_list_train_me=temp


from random import seed
from random import randint
seed(1)
temp=[]
y_test=[]
for i in ing_list_test_me:
    rand=randint(0, len(i)-1)
    temp2=" "
    for n,j in enumerate(i):
        if n!=rand:
            temp2+=(i[n]+" ")
        else:
            temp2+=("<mask> ")
            y_test.append(i[n])
    temp.append(temp2)
ing_list_test_me=temp

from pathlib import Path

from tokenizers import CharBPETokenizer

ing_sentence=temp

with open('ing_cat/ing_list_middleeast.txt', 'w') as fp:
    for i in ing_list_train_me:
        fp.write(i+"\n")


paths = [str(x) for x in Path(".").glob("ing_cat/ing_list_middleeast.*")]
tokenizer = CharBPETokenizer()
print(paths)
tokenizer.train("ing_cat/vocab.txt", vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>","</w>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("EsperBERTo_middleeast")


from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = CharBPETokenizer(
    "./EsperBERTo_middleeast/vocab.json",
    "./EsperBERTo_middleeast/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

print(tokenizer.encode("water butter sugar cinnamon nutmeg salt couscous almond golden_raisin date butter sugar <mask>"))
print(tokenizer.encode("water butter sugar cinnamon nutmeg salt couscous almond golden_raisin date butter sugar <mask>").tokens)

tokenizer.enable_truncation(max_length=512)


import torch

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo_middleeast", max_len=512)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)


model.num_parameters()


from transformers import DataCollatorForLanguageModeling

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="ing_cat/ing_list_middleeast.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./EsperBERTo_middleeast",
    overwrite_output_dir=True,
    num_train_epochs=150,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)
import torch
print(torch.__version__)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)
#trainer.train()

#trainer.save_model("./EsperBERTo_middleeast")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./EsperBERTo_middleeast",
    tokenizer="./EsperBERTo_middleeast"
)
predictions=[]
for i in ing_list_test_me: 
    predictions.append(fill_mask(i)[0]["token_str"])

for n,i in enumerate(predictions):
  print(predictions[n],y_test[n])

score=0
for n,i in enumerate(y_test):
    if predictions[n]==y_test[n]:
        score+=1
print(score,score/len(y_test))

checking={}
for i in predictions:
  if i not in checking:
    checking[i]=1
  else:
    checking[i]+=1

print(checking,len(checking))
real=y_test

with open("Data/categories_items_nospaces.json",encoding="utf8") as file:
  ing_cat_list=json.load(file)



def get_key(val):
    for key, value in ing_cat_list.items():
         if val in value:
             return key

inggg=[]
for i in real:
  temp=[]
  key=get_key(i)
  if key!=None:
    temp.append(key)
  inggg.append(temp)
real_cat=inggg




useless_words={}
inggg=[]
for i in predictions:
  temp=[]
  #print(j)
  key=get_key(i)
  #print(key)
  if key!=None:
    temp.append(key)
  else:
    if i not in useless_words:
      useless_words[i]=1
    else:
      useless_words[i]+=1
  inggg.append(temp)
print("yseless",len(useless_words))
pred_cat=inggg



print(real_cat[:10],pred_cat[:10])
scor_list=[]
for i, (r, p) in enumerate(zip(real_cat, pred_cat)):
  tem=[]
  for j in p:
    temp=[]
    for k in r:
      if j==k:
        temp.append(1)
      else:
        temp.append(0)
    tem.append(temp)
  scor_list.append(tem)

print(scor_list[:10])

fin_score_list=[]
for i in scor_list:
  temp_score=0
  for j in i:
    if 1 in j:
      temp_score+=1
  if len(i)!=0:
    fin_score_list.append(temp_score/len(i))
print(fin_score_list[:10])

print(sum(fin_score_list)/len(fin_score_list))

