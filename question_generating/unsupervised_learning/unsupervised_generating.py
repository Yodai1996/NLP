
# coding: utf-8

# In[196]:

import re, pickle, time
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
import MeCab
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn, optim
import torch.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# In[197]:

DATA_PATH="../../data/takken/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amp=200


# In[199]:

def remove_symbol(text):
    '''
    入力されたテキストから句読点などの不要な記号をいくつか削除する。
    '''
    remove_list = [
        ',', '.', '-', '，', '\ufeff', '\u3000', '\n']
    for i, symbol in enumerate(remove_list):
        text = text.replace(symbol, '')
    return text

def add_bos_eos(text):
    return "<BOS> " + text + " <EOS>"


# In[198]:

class Vocab():
    
    bos_token=2
    eos_token=3
    
    def __init__(self, min_count=0):
        self.word2id_dict = dict({'<PAD>':0,  '<UNK>': 1, '<BOS>':Vocab.bos_token, '<EOS>':Vocab.eos_token})
        self.id2word_dict = dict({i:word for word, i in self.word2id_dict.items()})
        self.size = 4
        self.min_count = min_count
        self._i = 0  
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i==self.size:
            self._i=0
            raise StopIteration
        word = self.id2word(self._i)
        self._i += 1
        return word
        
    def add(self, word):
        key = self.word2id_dict.setdefault(word, self.size)
        self.id2word_dict[key] = word
        if key == self.size:
            self.size += 1      
            
    def id2word(self, key):
        return self.id2word_dict.get(key)
    
    def word2id(self, key):
        return self.word2id_dict.get(key)
    

    
#batch_size and datasize are global variables
class DataLoader():
    def __init__(self, inputs):
        self.start_index = 0
        self.inputs=inputs
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start_index >= datasize:
            self.start_index =0 
            raise StopIteration
        minibatch = self.inputs[self.start_index: self.start_index+batch_size]
        self.start_index += batch_size
        
        minibatch = torch.tensor(minibatch, dtype=torch.long, device=device)
        return minibatch
    
    
    
def trainIters(model, criterion, dataloader, words, print_every=1, plot_every=5,):
    optimizer = optim.Adam(model.parameters())
    plot_losses = []
    for epoch in range(epochs):
        start=time.time()
        train_loss=0
        for batch_id, minibatch in enumerate(dataloader):
            a   = minibatch[:,:-1][:]
            generated = model(a)
            hoge = minibatch[:,1:][:]

            loss = criterion(generated, hoge)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            if batch_id % print_every == 0:
                elapsed_sec = time.time() - start
                elapsed_min = int(elapsed_sec / 60)
                elapsed_sec = elapsed_sec - 60 * elapsed_min
                print(
                    'Epoch:{} Batch:{}/{} Loss:{:.4f} Time:{}m{:.1f}s'.format(
                        epoch, batch_id,
                        int(datasize /
                            batch_size),
                        train_loss / (1 + batch_id), elapsed_min,
                        elapsed_sec),
                    end='\r')
                
            if batch_id % plot_every:
                plot_losses.append(loss)
        print()            
        
    return plot_losses


# In[215]:

class Rnn(nn.Module):
    def __init__(self, latent_dim, embedding_matrix):
        super(Rnn, self).__init__()
        self.lstm          = nn.LSTM(input_size = embedding_dim, hidden_size = latent_dim)
        self.linear        = nn.Linear(latent_dim, num_tokens)
        embedding_matrix   = torch.from_numpy(embedding_matrix)
        self.embed         = nn.Embedding(num_tokens, embedding_dim=embedding_dim, padding_idx=0, _weight=embedding_matrix)
        
    def forward(self, inputs):
        #以下，maxlenはその値より-1したものを表す
        
        #inputs = (batch, maxlen)
        #inputsはindexに対応
        batch_size     = inputs.shape[0]
        embed          = self.embed(inputs) #(batch, maxlen, embedding_dim)
        
        #make random states
        h=torch.rand(1,batch_size,latent_dim)*amp
        c=torch.rand(1,batch_size,latent_dim)*amp
        states = (h.cuda(), c.cuda())
        
        outputs, hidden = self.lstm(
            embed.permute(1,0,2), states
        ) #(maxlen, batch, latent_dim)
        outputs = self.linear(outputs)  #(maxlen, batch, num_tokens)
        outputs = outputs.permute(1,0,2) #(batch, maxlen, num_tokens)
        outputs = outputs.permute(0,2,1) #(batch, num_tokens, maxlen)
         
        return outputs
    
    
    def sample(self, words, size=20):
        outputs=[]
        
        #describe initial input and initial random states  
        income = torch.tensor(
            [words.word2id("<BOS>")]*size, dtype=torch.long, device=device).unsqueeze(1)  #(size, 1)
        
        h=torch.rand(1, size, latent_dim)*amp
        c=torch.rand(1, size, latent_dim)*amp
        states = (h.cuda(), c.cuda())        
        
        #回す(ほんとは<EOS>で止めるようにしたかった。)
        for i in range(maxlen-1):     
            embed = self.embed(income) #(size, 1, embedding_dim)
            embed = embed.permute(1,0,2)  #(1, size, embedding_dim)
            
            outcome, hidden = self.lstm(embed, states)
            outcome = self.linear(outcome)  #(1, size, num_tokens)
            _, topi = torch.max(outcome, 2)
            outputs.append(topi)
            income = topi.permute(1,0)  #(size, 1)
            
            states = hidden
            
        outputs = torch.cat(outputs, dim=0).permute(1,0)  #(size, max_len)
        
        return outputs            


# In[200]:

takken    = pd.read_csv(DATA_PATH+"takken.csv", encoding='utf-8')
mondaishu = pd.read_csv(DATA_PATH+"mondaishu.csv", encoding='utf-8')
nikken    = pd.read_csv(DATA_PATH+"nikken.csv", encoding='utf-8')
legal_mind= pd.read_csv(DATA_PATH+"legal_mind.csv", encoding='utf-8')

takken=takken[["Question","Choice"]]
ocr=pd.concat([mondaishu, nikken, legal_mind], axis=0, ignore_index=True)
ocr = ocr[["Wakati_Question", "Wakati_Choice"]]
ocr.columns = ["Question", "Choice"]

m=MeCab.Tagger("-Owakati")
takken = takken.applymap(remove_symbol)
ocr    = ocr.applymap(remove_symbol)
takken = takken.applymap(m.parse)
takken = pd.concat([takken, ocr], axis=0, ignore_index=True)
takken = takken.applymap(remove_symbol)
takken = takken.applymap(add_bos_eos)


# In[201]:

takken=takken["Choice"]
takken


# In[202]:

words=Vocab()
for sent in takken:
    for word in sent.split():
        words.add(word)


# In[203]:

num_tokens=words.size


# In[204]:

word2vec = Word2Vec.load("../wiki_textbook/text_wiki_model")
word2vec_size = 200
embedding_dim = word2vec_size
embedding_matrix = np.random.uniform(low=-0.1, high=0.1, size=(num_tokens, word2vec_size))


# In[205]:

unknowns=set()

for i,word in enumerate(words):
    try:
        embedding_matrix[i] = word2vec[word]
    except KeyError:
        unknowns.add(word)
        
embedding_matrix[0] = np.zeros((word2vec_size, ))

embedding_matrix = embedding_matrix.astype('float32')


# In[206]:

unknowns.remove("<BOS>")
unknowns.remove("<EOS>")
unknowns.remove("<UNK>")
unknowns.remove("<PAD>")
unknowns


# In[207]:

datasize = len(takken)


# In[208]:

maxlen=0
for sent in takken:
    length= len(sent.split())
    if length>maxlen:
        maxlen=length


# In[209]:

#<BOS>付き
maxlen


# In[210]:

inputs=np.zeros((datasize, maxlen), dtype='int32')


# In[211]:

for i,sent in enumerate(takken):
    for j,word in enumerate(sent.split()):
        if word in unknowns:
            word="<UNK>"
        inputs[i][j]=words.word2id(word)


# In[213]:

criterion=nn.CrossEntropyLoss(ignore_index=0) 
latent_dim=512
batch_size=32
epochs = 28


# In[214]:

embedding_matrix.shape[0]


# In[220]:

model=Rnn(latent_dim, embedding_matrix).to(device)
dataloader = DataLoader(inputs)
losses = trainIters(model, criterion, dataloader, words)


# In[221]:

size=50

generated = model.sample(words, size) #(size, maxlen)
kinds = set()
generated = generated.tolist()
for sample in generated:
    sent=""
    
    for index in sample:
        #When <EOS>, finish. This will make sense.
        if index==words.word2id("<EOS>"):
            break       
                
        word = words.id2word(index)
        sent =sent + word + " "
        print(word, end=" ")
    
    print(end="\n\n")
    kinds.add(sent)


# In[222]:

print("different kinds of sentences in %d is %d" % (size, len(kinds)))

