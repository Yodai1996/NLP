
# coding: utf-8

# In[162]:

import glob
import xml.etree.ElementTree as ET
import re
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import json


# In[266]:

tagger =MeCab.Tagger("-Ochasen")
year = 2018
input_path='/home/develop/data/qualification/takken/kyozai/quizzes/quiz_%d.csv' % year
xml_dir ='/home/develop/user/k_satou/textproc/%d/' % year

#the size of feature words used in each chapter in textbook.
extract_size = 40


#this score means how high you will value numbers information 
#In the preprocess for tfidf, we delete information such as numbers
#so if you want to use such information in order to make labeling with higher precision,
#you should use the json data of title2numbers.
use_number=True
number_score=0.5

#Same to index information, which is not used in tfidf only.
#use title2indexwords.xml
use_index=True
index_score=0.5


# In[267]:

def reshape(sentences):
    sentences = [re.sub(' |\u3000|_|　', ' ', sent) for sent in sentences]
    return sentences

def tokenize(sentence):
    node=tagger.parse(sentence)
    node=node.split("\n")
    tokenized_sentence=[]
    for s in node:
        feature=s.split("\t")
        if feature[0]=="EOS":
            break
        
        #以下は，何を前処理として消すか，という話なので，工夫の余地あり.
        #number is ignored
        elif feature[3]=="名詞-数":
            pass      
        elif feature[3]=="記号-空白":
            pass
        elif feature[3]=="助動詞":
            pass
        elif feature[0]=="-":
            pass
        elif feature[0]=="/":
            pass
        elif feature[0]=="▪":
            pass
        elif feature[0]=="I":
            pass
        elif feature[0]=="II":
            pass
        elif feature[0]=="III":
            pass
        elif feature[0]=="IV":
            pass
        elif feature[0]=="V":
            pass 
        elif feature[0]=="VI":
            pass        
        elif feature[0]=="check":
            pass
        elif feature[0]=="さん":
            pass
        elif feature[0]=="こちら":
            pass
        elif feature[0]=="よう":
            pass        
        elif feature[0]=="ため":
            pass
        elif feature[0]=="かたち":
            pass     
        elif feature[0]=="こと":
            pass 
        elif '❶' in feature[0]:
            pass
        elif feature[0]=="どおり":
            pass
        #名詞以外の品詞については，重要度が低めなので取り除いてみた.
        elif feature[3][0:2]!="名詞":
            pass
        else:
            tokenized_sentence.append(feature[0])
    
    #space区切りのstr型にする
    tokenized_sentence = " ".join(tokenized_sentence)
    
    return tokenized_sentence


# In[282]:

def calculate(raw_input, input_dic, titles, title2keywords, use_number=use_number, use_index=use_index):

    title2score = {}
    for title in titles:
        keyword2score = title2keywords[title]
        total_score=0
        
        if use_number:
            numbers = title2numbers[title]
            for number in numbers:
                if number in raw_input:
                    total_score += number_score
                    
        if use_index:
            if title in title2indexwords:
                indices = title2indexwords[title]
                for ind in indices:
                    if ind in raw_input:
                        total_score += index_score

        for keyword, score in keyword2score.items():
            if keyword in input_dic:
                total_score += input_dic[keyword] * score
                
        title2score[title] = total_score
        
    #return only max
    max_score=0
    max_title=0
    for title, score in title2score.items():
        if score > max_score:
            max_score = score
            max_title = title
            
    return max_title, max_score  


# In[268]:

node=tokenize("明日いい天気になれ")
node


# In[269]:

xmls = sorted(glob.glob(xml_dir + '*Text*.xml'))
xml  = xmls[0]


# In[270]:

title2page = {}
title2keywords={}


# In[271]:

tree = ET.parse(xml)
root = tree.getroot()
parent_map = {c:p for p in root.iter() for c in p}
texts=[]
for c in parent_map:
    if c.tag=="text":
        title = parent_map[c].attrib["title"]
        page  = int(parent_map[c].attrib["from_page"])
        title2page[title]=page
        
        text=c.text
        processed_text = tokenize(text)
        texts.append(processed_text)


# #####  carry out tdidf in textbook

# In[272]:

vectorizer = TfidfVectorizer(use_idf=True)

#2次元行列。（tfidfのスコアが中身）
tfidf_matrix = vectorizer.fit_transform(texts)
tfidf_matrix = tfidf_matrix.toarray()

#各ラベル名（単語名）を入手
terms = vectorizer.get_feature_names()

#辞書にkeywordsとscoresを追加しよう
for i,title in enumerate(title2page.keys()):
    tfidf_array  = tfidf_matrix[i]
    tfidf_scores = np.sort(tfidf_array)[-extract_size:][::-1].tolist()       
    top_n_index  = tfidf_array.argsort()[-extract_size:][::-1]
    words = [terms[index] for index in top_n_index]

    #辞書を作成する.
    word2score={}
    for i,word in enumerate(words):
        word2score[word]=tfidf_scores[i]
        
    #メタ辞書にこの辞書を追加する
    title2keywords[title] = word2score


# ##### Inputs

# In[276]:

df=pd.read_csv(input_path)


# In[278]:

df_extracted=df["問題文"]+df["解説"]
inputs = df_extracted.tolist()
courses= df["講座"].tolist()

courses = reshape(courses)
courses


# ##### 講座のコラムをもとに調べる項を限定する

# In[279]:

dfr = pd.read_csv("./lesson2page.csv")
lessons = dfr["lesson"].tolist()
starts=dfr["page_start"].tolist()
ends  =dfr["page_end"].tolist()

lessons = reshape(lessons)
lessons


# ##### carry out tfidf also in inputs

# In[ ]:

tfidf_inputs = [tokenize(sent) for sent in inputs]
processed_inputs=[]

#2次元行列。（tfidfのスコアが中身）
tfidf_matrix = vectorizer.fit_transform(tfidf_inputs)
tfidf_matrix = tfidf_matrix.toarray()

#各ラベル名（単語名）を入手
terms = vectorizer.get_feature_names()

for i in range(len(tfidf_matrix)):
    tfidf_array  = tfidf_matrix[i]
    dic={}
    for j,word in enumerate(terms):
        if tfidf_array[j] > 0:
            dic[word] = tfidf_array[j]
    processed_inputs.append(dic)


# In[ ]:

with open("./data/title2numbers.json") as f:
    title2numbers = json.load(f)
f.close()

with open("./data/title2indexwords.json") as f:
    title2indexwords = json.load(f)
f.close()


# In[285]:

ans = []

assert len(courses)==len(inputs)

for i,course in enumerate(courses):
    raw_input = inputs[i]
    input_dic = processed_inputs[i]
    for j,lesson in enumerate(lessons):
        #↓一致したら，という気分
        if lesson in course:
            start = starts[j]
            end   = ends[j]

            related_titles=[]
            for title,page in title2page.items():
                #↓範囲指定の区切りが美しい，という前提の下
                if start<=page and page<=end:
                    related_titles.append(title)

            best_title, best_score = calculate(raw_input, input_dic, related_titles, title2keywords)

            #入力もtfidfしようか
            ans.append(best_title)

            break   
            

for title in ans:
    print(title2page[title], end=" ")
    print(title)

