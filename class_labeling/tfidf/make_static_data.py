
# coding: utf-8

# In[85]:

import unicodedata
import glob
import xml.etree.ElementTree as ET
import re
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import json
import bisect


# In[86]:

year = 2018
csv_path='/home/develop/data/qualification/takken/kyozai/quizzes/quiz_%d.csv' % year
xml_dir ='/home/develop/user/k_satou/textproc/%d/' % year


# In[87]:

xml_paths = sorted(glob.glob(xml_dir + '*Text*.xml'))
print(xml_paths[0])


# In[88]:

#前処理用の関数
def preprocessing(text):
    text = text.replace('\n','')
    text = text.replace(' ','')
    text = text.replace('　','')
    normalize(text)
    return(text)

def normalize(text):
    text = normalize_unicode(text)
    text = lower_text(text)
    return text

def lower_text(text):
    #小文字に統一
    return text.lower()

def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, text)
    return normalized_text


# In[97]:

def pickup_title(page2title, page):
    start_pages = list(page2title.keys())
    insert_index = bisect.bisect_right(start_pages, page)
    start_page = start_pages[insert_index - 1]
    return page2title[start_page]


# In[89]:

#全テキストから辞書作成
path2page2title = {}
title2keywords = {}
title2numbers = {}
number2titles = {}
all_numbers = set()

title2indexwords={}
indexword2titles={}
all_titles=set()

for i, xml_path in enumerate(xml_paths):
    tree = ET.parse(xml_paths[i])
    root = tree.getroot()
    parent_map = {c:p for p in root.iter() for c in p}
    page2title={}
    for c in parent_map:
        if c.tag=="text":
            title = parent_map[c].attrib["title"]
            page  = int(parent_map[c].attrib["from_page"])
            page2title[page]=title
            text=preprocessing(c.text)
            k = re.findall(r"[0-9]+項", text)
            m = re.findall(r"[0-9]+条", text)
            n = re.findall(r"[0-9]+年", text)
            l = re.findall(r"[0-9]+m", text)
            o = re.findall(r"[0-9]+m2", text)
            p = re.findall(r"[0-9]+階", text)
            q = re.findall(r"[0-9]+日", text)
            title2numbers[title] = set(k) | set(m)|set(n)|set(l)|set(o)|set(p)|set(q)
            #recursive
            all_numbers = all_numbers | title2numbers[title]
    path2page2title[xml_path] = page2title
print(all_numbers)
print("\n\n")

for number in all_numbers:
    titles = [title for title, numbers in title2numbers.items() if number in numbers]
    number2titles[number] = titles
print(title2numbers)
print("\n\n")
print(number2titles)


# In[90]:

f = open("./data/number2titles.json", "w")
json.dump(number2titles, f, ensure_ascii=False, indent=1)
f.close()

f = open("./data/path2page2title.json", "w")
json.dump(path2page2title, f, ensure_ascii=False, indent=1)
f.close()


t2n={}
for k,v in title2numbers.items():
    v=list(v)
    t2n[k]=v
    
f = open("./data/title2numbers.json", "w")
json.dump(t2n, f, ensure_ascii=False, indent=1)
f.close()


# In[105]:

indices_csv=["index1.csv", "index2.csv", "index3.csv","index4.csv"]
for i,csv_file in enumerate(indices_csv):
    df=pd.read_csv("../index_data/"+csv_file)
    df=df[["nomble", "word"]]
    xml_path = xml_paths[i]
    page2title = path2page2title[xml_path]
    for j, word in enumerate(df["word"].tolist()):
        titles=set()
        #there are some words that have 2 or more relative pages, so make list of pages
        pages = df["nomble"][j]
        pages = str(pages).split(",")
        for page in pages:
            page=int(page)
            title = pickup_title(page2title, page)
            #print(title)
            titles.add(title)
            all_titles.add(title)
        indexword2titles[word]=titles     

        
for title in all_titles:
    words = [word for word, titles in indexword2titles.items() if title in titles]
    title2indexwords[title]=words


# In[106]:

indexword2titles["事務所"]


# In[108]:

title2indexwords['「事務所」の意義']


# In[109]:

f = open("./data/title2indexwords.json", "w")
json.dump(title2indexwords, f, ensure_ascii=False, indent=1)
f.close()

