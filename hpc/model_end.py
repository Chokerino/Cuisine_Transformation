
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
from keras.models import model_from_json

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

cook_list=["and","drop","ladle","with","heat","cook","remove","cool","blend","push","stir",'saute', 'evaporated',"melt","store","foil","press","dry","put","break","seal","top","mash","toast","reduce","crush","tablespoon","break","crush","beat","sift","knife","board","set","divide","combine","toss","seperate","fry","bowl","fork","cut","sheet","season","oven","scoop","slice","broil","pierce","wash","dressing","whisk","coat","refrigerate","coat","preheat","transfer","chill","roll","cup","spoon","saucepan","smooth","pour","stirring","add","boil","simmer","pot","fold","skillet","pan","spatula","cover","soak","drain","serve","mix","place",'knead','separate','spread','bake','sheet','sheet','process','drizzle','processor','chop','sprinkle','garnish','peel','taste','blender',"dish"]

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
    temp.append(j.replace(" ","_"))
  temp1.append(temp)
ing_list_cleaned=temp1

"""modify for training"""

import random
ing_list_input=[]
ing_list_output1=[]
ing_list_output2=[]
ing_list_output3=[]
cuisines_final=[]
recipe_names_final=[]
for n, i in enumerate(ing_list_cleaned):
  i=set(i)
  if(len(i)>=7):
    b = set(random.sample(i, len(i)-3))  # the `if i in b` on the next line would benefit from b being a set for large lists
    c = [j for j in i if j not in b]
    ing_list_input.append(list(b))
    ing_list_output1.append(c[0])
    ing_list_output2.append(c[1])
    ing_list_output3.append(c[2])
    cuisines_final.append(cuisines[n])
    recipe_names_final.append(recipe_names_corrected[n])

print(len(ing_list_output1),
len(ing_list_input),
len(cuisines_final))

for i in recipe_names_final:
      ing_list_cleaned.append(i)

"""Assign Label to Cuisines"""

dict_cui={}
x=0

count_cui={}
for i in cuisines_final:
  if i not in dict_cui:
    dict_cui[i]=x
    count_cui[i]=1
    x+=1
  else:
    count_cui[i]+=1
print(dict_cui)

for n,i in enumerate(cuisines_final):
  cuisines_final[n]=dict_cui[i]
print(count_cui)



"""Unique Ings.(No need to run)"""

"""Word2Vec"""

import gensim 

EMBEDDING_DIM = 100
# train word2vec model
model = gensim.models.Word2Vec(sentences=ing_list_cleaned, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
# vocab size
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))
#print(words)


temp=[]
for i in recipe_names_final:
    temp2=[]
    for j in i:
        if j != "and" and j!= "with" and j!= "the":
            temp2.append(j)
    temp.append(j)
recipe_names_final=temp
#print(recipe_names_final)	

embeddings_index = {}
for i in words:
  embeddings_index[i]=model[i]
embeddings_index

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

max_length=100
# vectorize the text samples into a 2D integer tensor
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(ing_list_cleaned)
sequences = tokenizer_obj.texts_to_sequences(ing_list_input)
sequences2 = tokenizer_obj.texts_to_sequences(ing_list_output1)
sequences2a = tokenizer_obj.texts_to_sequences(ing_list_output2)
sequences2b = tokenizer_obj.texts_to_sequences(ing_list_output3)

sequences3= tokenizer_obj.texts_to_sequences(recipe_names_final)

# pad sequences
word_index = tokenizer_obj.word_index
print('Found %s unique tokens.' % len(word_index))

ing_list_converted_input = pad_sequences(sequences, maxlen=max_length)
ing_list_converted_output_1 = pad_sequences(sequences2, maxlen=max_length)
ing_list_converted_output_2 = pad_sequences(sequences2a, maxlen=max_length)
ing_list_converted_output_3 = pad_sequences(sequences2b, maxlen=max_length)
recipe_names_input=pad_sequences(sequences3,maxlen=max_length)

print('Shape of review tensor:', ing_list_converted_input.shape)
#print('Shape of review tensor:', ing_list_converted_output.shape)

print(np.asarray(cuisines).shape)

X_train_1, X_validate_1, X_test_1 = np.split(ing_list_converted_input, [int(.6*len(ing_list_converted_input)), int(.8*len(ing_list_converted_input))])
len(X_train_1)

y_train_1, y_validate_1, y_test_1 = np.split(ing_list_converted_output_1, [int(.6*len(ing_list_converted_output_1)), int(.8*len(ing_list_converted_output_1))])
#y_train.shape

y_train_2, y_validate_2, y_test_2 = np.split(ing_list_converted_output_2, [int(.6*len(ing_list_converted_output_2)), int(.8*len(ing_list_converted_output_2))])
#y_train.shape

y_train_3, y_validate_3, y_test_3 = np.split(ing_list_converted_output_3, [int(.6*len(ing_list_converted_output_3)), int(.8*len(ing_list_converted_output_3))])
#y_train.shape


from keras.utils import to_categorical
X_train_2, X_validate_2, X_test_2 = np.split(to_categorical(np.asarray(cuisines_final),num_classes=200), [int(.6*len(to_categorical(np.asarray(cuisines_final),num_classes=200))), int(.8*len(to_categorical(np.asarray(cuisines_final),num_classes=200)))])
X_train_2.shape

X_train_3, X_validate_3, X_test_3 = np.split(recipe_names_input, [int(.6*len(recipe_names_input)), int(.8*len(recipe_names_input))])
len(X_train_3)

EMBEDDING_DIM =100
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.models import Sequential
from keras.layers import Multiply,Activation,Dense, Embedding, LSTM, GRU,RepeatVector,TimeDistributed,Concatenate,Input,Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.models import Model
import keras.backend as K
# define model



json_file = open('models/model_end.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("models/model_end.h5")
print("Loaded model2 from disk")

import sys
import tensorflow as tf
np.set_printoptions(threshold=1000)
z,z_1,z_2=model.predict([X_test_1,X_test_2])

print(z[:10])
temp=[]
for i in z:
    for j in embeddings_index:
        if np.allclose(embeddings_index[j],i):
            temp.append(j)
z=temp

print("hehehe",z[:10])
for i in z_1:
    for j in embeddings_index:
        if pad_sequences(embeddings_index[j], maxlen=max_length)==i:
            temp.append(j)
z_1=temp

for i in z_2:
    for j in embeddings_index:
        if pad_sequences(embeddings_index[j], maxlen=max_length)==i:
            temp.append(j)
z_2=temp












temp=[]
for i,j in enumerate(z):
    temp2=[]
    temp2.append(z_1[i])
    temp2.append(z_2[i])
    temp.append(temp2)
#z=temp

temp=[]
for i,j in enumerate(y_test_1):
    temp2=[]
    temp2.append(j)
    temp2.append(y_test_2[i])
    temp2.append(y_test_3[i])
    temp.append(temp2)
y_test=temp


print(z[0])

#y_test[0]

#print(list(tokenizer_obj.sequences_to_texts(z.astype(int))))


predictions_1=list(tokenizer_obj.sequences_to_texts(z.astype(int)))
predictions_2=list(tokenizer_obj.sequences_to_texts(z_1.astype(int)))
predictions_3=list(tokenizer_obj.sequences_to_texts(z_2.astype(int)))

print(predictions_1[:10],predictions_2[:10],predictions_3[:10])

print("\n",list(tokenizer_obj.sequences_to_texts(y_train_1))[:10])
import json
#with open('dumps/dump_fix_p.json', 'w', encoding='utf-8') as f:
#    json.dump(predictions,f)

#with open('dumps/dump_fix_r.json', 'w', encoding='utf-8') as f:
#    json.dump(list(tokenizer_obj.sequences_to_texts(y_test)),f)

#with open('dump_fix_name.json', 'w', encoding='utf-8') as f:
#    json.dump(list(tokenizer_obj.sequences_to_texts(X_test_3)),f)

#with open('dump_fix_inglist.json', 'w', encoding='utf-8') as f:
#    json.dump(list(tokenizer_obj.sequences_to_texts(X_test_1)),f)

real_1=list(tokenizer_obj.sequences_to_texts(y_test_1))
real_2=list(tokenizer_obj.sequences_to_texts(y_test_2))
real_3=list(tokenizer_obj.sequences_to_texts(y_test_3))

temp=[]
for i,j in enumerate(predictions_1):
    temp2=""
    temp2+=(j+" ")
    temp2+=(predictions_2[i]+" ")
    temp2+=(predictions_3[i])
    temp.append(temp2)
predictions=temp

temp=[]
for i,j in enumerate(real_1):
    temp2=""
    temp2+=(j+" ")
    temp2+=(real_2[i]+" ")
    temp2+=(real_3[i])
    temp.append(temp2)
real=temp


rnamess=list(tokenizer_obj.sequences_to_texts(X_test_3))
ings=list(tokenizer_obj.sequences_to_texts(X_test_1))

with open("Data/categories_items.json",encoding="utf8") as file:
  ing_cat_list=json.load(file)


q=[]
for i in predictions:
  j=i.split(" ")
  q.append(j)
 # print(j)
predictions=q

q=[]
for i in real:
  j=i.split(" ")
  q.append(j)
 # print(j)
real=q

temp_dict={}
for i in predictions:
  for j in i:
    if j not in temp_dict:
      temp_dict[j]=0
    else:
      temp_dict[j]+=1
print(len(temp_dict))
print(temp_dict)

def get_key(val): 
    for key, value in ing_cat_list.items(): 
         if val in value: 
             return key 

inggg=[]
for i in real:
  temp=[]
  for j in i:
    #print(j)
    key=get_key(j)
    #print(key)
    if key!=None:
      temp.append(key)
  inggg.append(temp)
real_cat=inggg

useless_words={}
inggg=[]
for i in predictions:
  temp=[]
  for j in i:
    #print(j)
    key=get_key(j)
    #print(key)
    if key!=None:
      temp.append(key)
    else:
      if j not in useless_words:
        useless_words[j]=1
      else:
        useless_words[j]+=1
  inggg.append(temp)
pred_cat=inggg
print(useless_words)

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

fin_score_list=[]
for i in scor_list:
  temp_score=0
  for j in i:
    if 1 in j:
      temp_score+=1
  if len(i)!=0:
    fin_score_list.append(temp_score/len(i))
fin_score_list
print(sum(fin_score_list)/len(fin_score_list))
