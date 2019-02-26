
# coding: utf-8

# シェルスクリプトでいくつかの実験設定で動かすにはコマンドライン引数を扱えるようにしていく必要がある。引数を辞書型に格納できると楽かも。
# 
# EncoderDecoderで一つのクラスにまとめる。(forwardメソッドの挙動を追う)
# 
# 

# In[2]:

import pickle
import re
import time
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
import MeCab
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
from torch import optim
import torch.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


# In[212]:

DATA_PATH="./data/"
MODEL_PATH="./saved_model/"
MODEL_FILE="encoderdecoder_correct.model"
LOSS_FILE="encoderdecoder.png"
SAMPLE_FILE="encoderdecoder_correct_sample.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
batch_size = 32
latent_dim = 512  # Latent dimensionality of the encoding space.


# In[249]:

class Vocab(object):
    '''単語とIDのペアを管理するクラス。
    Attributes:
        min_count: 未実装，min_count以下の出現回数の単語はVocabに追加しないようにする
        
    TODO:
        add min_count option
    '''
    
    bos_token = 2
    eos_token = 3

    def __init__(self, min_count=0):
        self.word2id_dict = dict({'<PAD>': 0, '<UNK>': 1, '<BOS>':Vocab.bos_token, '<EOS>':Vocab.eos_token})
        self.id2word_dict = dict(
            {i: word
             for word, i in self.word2id_dict.items()})
        self.size = 4
        self.min_count = min_count
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self.size:
            self._i = 0
            raise StopIteration
        word = self.id2word(self._i)
        self._i += 1
        return word

    def add(self, word):
        '''
        Args:
            word(string):単語
        if word is not in Vocab, then add it
        '''
        key = self.word2id_dict.setdefault(word, self.size)
        self.id2word_dict[key] = word
        if key == self.size:
            self.size += 1
            
    def delete(self, word):
        try:
            key = self.word2id_dict.pop(word)
            self.id2word_dict.pop(key)
            self.size -= 1
        except KeyError:
            print('{} doesn\'t exist'.format(word))
        

    def word2id(self, word):
        '''
        Args:
            word(string):単語
        Returns:
            returns id allocated to word if it's in Vocab. Otherwise, returns 1 which means unknown word.
        '''
        return self.word2id_dict.get(word, 1)  #1 means <UNK>

    def id2word(self, key):
        '''
        Args:
            key(int)
        Returns:
            returns word allocated to key if it's in Vocab. Otherwise, returns <UNK>.
        '''
        return self.id2word_dict.get(key, '<UNK>')

    def build_vocab(self, sentences):
        '''update vocab
        Args:
            sentences:list of lists,each element of list is one sentence,
            each sentence is represented as list of words
        '''
        assert isinstance(sentences, list)

        for sentence in sentences:
            assert isinstance(sentence, list)
            for word in sentence:
                self.add(word)

    def seq2ids(self, sentence):
        '''
        Args:
            sequence: list each element of which is word(string)
        Returns:
            list each element of which is id(int) corresponding to each word
        '''
        assert isinstance(sentence, list)
        id_seq = list()
        for word in sentence:
            id_seq.append(self.word2id(word))

        return id_seq

    def ids2seq(self, id_seq):
        '''inverse processing of seq2ids
        '''
        assert isinstance(id_seq, list)
        sentence = list()
        for key in id_seq:
            sentence.append(self.id2word(key))
            if sentence[-1] == '<EOS>':
                break
        return sentence


class DataLoader(object):
    '''Data loader to return minibatches of input sequence and target sequence an iteration
    Attributes:
        input_seq: input sequence, numpy ndarray
        target_seq: target sequence, numpy ndarray
        input_lengths: true lengths of input sequences, before padding
        batch_size: batch size
    '''

    def __init__(self, src_seq, tgt_seq, src_lengths, batch_size):
        self.src_seq = src_seq
        self.tgt_seq = tgt_seq
        self.src_lengths = src_lengths
        self.batch_size = batch_size
        self.size = len(self.src_seq)
        self.start_index = 0
        self.reset()

    def reset(self):
        '''shuffle data'''
        self.src_seq, self.tgt_seq, self.src_lengths = shuffle(
            self.src_seq, self.tgt_seq, self.src_lengths)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_index >= self.size:
            self.reset()
            self.start_index = 0
            raise StopIteration
        batch_X = self.src_seq[self.start_index:self.start_index +
                               self.batch_size]
        batch_Y = self.tgt_seq[self.start_index:self.start_index +
                               self.batch_size]
        lengths = self.src_lengths[self.start_index:self.start_index +
                                   self.batch_size]
        self.start_index += self.batch_size

        batch_X = torch.tensor(batch_X, dtype=torch.long, device=device)
        batch_Y = torch.tensor(batch_Y, dtype=torch.long, device=device)
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)

        lengths, perm_idx = lengths.sort(descending=True)
        batch_X = batch_X[perm_idx]
        batch_Y = batch_Y[perm_idx]        
        return batch_X, batch_Y, lengths


class BiEncoder(nn.Module):
    '''Bidirectional Encoder
    Attributes:
        num_vocab: vocabulary size of input sequences
        embedding_dim: dimensions of embedding vector
        hidden_size: hidden dimensions of LSTM
        embedding_matrix: initial values of embedding matrix
    '''

    def __init__(self, num_vocab, embedding_dim, hidden_size,
                 embedding_matrix):
        super(BiEncoder, self).__init__()
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            num_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_matrix)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True)

    def forward(self, x, lengths):
        '''
        Args:
            x: input sequence (batch_size, seq_len)
            lengths: tensor that retains true lengths before padding
        Returns:
            output: LSTM output
            (h, c): LSTM states at last timestep
        '''
        """
        embed = self.embed(x).permute(1, 0, 2).to(
            device)  #(seq_len, batch_size, embedding_dim)
        """
        embed = self.embed(x)  #(batch_size, seq_len, hidden_size)
        
        
        embed = pack_padded_sequence(
            embed, lengths=lengths, batch_first=True
        )  #(any_len, batch_size, embedding_dim),もう少し調べてから実装する
        assert embed[0].size(0) == torch.sum(lengths), '{},{}'.format(
            embed[0].size(0), torch.sum(lengths))        

        output, (h, c) = self.bilstm(
            embed
        )  #(seq_len, batch_size, 2*hidden_size), (2, batch_size, hidden_size)
        # reshape states into (1,batch_size, 2*hidden_size)
        h = h.permute(1, 2, 0).contiguous().view(1, -1, 2 * self.hidden_size)
        c = c.permute(1, 2, 0).contiguous().view(1, -1, 2 * self.hidden_size)


        
        output, lengths = pad_packed_sequence(
            output)  #(seq_len, batch_size, hidden_size)
        #print("2touple size = %d" %len(output))
        

        return output, (h, c) #(seq_len, batch_size, hidden_size)


class Decoder(nn.Module):
    '''NN decoding from encoder's last states
    Args:
        num_vocab: vocabulary size of target sequences
        embedding_dim: dimensions of embedding vector
        hidden_size: hidden dimensions of LSTM
        embedding_matrix: initial values of embedding matrix
    '''

    def __init__(self, num_vocab, embedding_dim, hidden_size,
                 embedding_matrix):
        super(Decoder, self).__init__()
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(
            num_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_matrix)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, num_vocab)

    def forward(self, decoder_input, decoder_states):
        '''
        Args:
            decoder_input: tensor (batch_size, seq_len)
            decoder_states: LSTM's initial state (1, batch_size, hidden_dim)
            
        Returns:
            output: LSTM output shape=(seq_len,batch_size,num_vocab)
            hidden: tuple of last states, both shape=(1,batch_size,hidden_dim)
        '''
        embed = self.embed(decoder_input)  #(batch_size,seq_len,embedding_dim)
        assert len(embed.size()) == 3, '{}'.format(embed.size())
        output, hidden = self.lstm(
            embed.permute(1, 0, 2), decoder_states
        )  #(seq_len,batch_size,hidden_dim),(1,batch_size,hidden_dim)
        output = self.linear(output)  #(seq_len,batch_size,num_vocab)

        return output, hidden  # (seq_len,batch_size,num_vocab), tuple of (1,batch_size,hidden_dim)


class LocalAttentionDecoder(nn.Module):
    '''Decoder using Global Attention mechanism
     Args:
        num_vocab: vocabulary size of target sequences
        embedding_dim: dimensions of embedding vector
        hidden_size: hidden dimensions of LSTM
        maxlen: maximum length of input sequences
        embedding_matrix: initial values of embedding matrix
        dropout_p: probability of dropout occurrence, Default:0.2
    '''

    def __init__(self,
                 num_vocab,
                 embedding_dim,
                 hidden_size,
                 maxlen,
                 embedding_matrix,
                 dropout_p=0.2,
                 d_size=20):
        super(LocalAttentionDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.maxlen = maxlen
        self.dropout_p = dropout_p
        embedding_matrix = torch.from_numpy(embedding_matrix)
        self.embedding = nn.Embedding(
            num_vocab,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
            _weight=embedding_matrix)
        #self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=self.hidden_size)
        self.out = nn.Linear(2 * hidden_size, num_vocab)
        self.linear = nn.Linear(hidden_size, 1)
        self.d_size = d_size
        self.num_vocab = num_vocab

    def forward(self, decoder_input, hidden, encoder_outputs, mask=None):
        '''
        Args:
            decoder_input: (batch_size, seq_len),seq_len must be 1
            hidden: LSTM initial state, tuple(h,c)
            encoder_outputs: (seq_len,batch_size,hidden_size)
        Returns:
            output: LSTM output
            hidden: LSTM last states
            attn_weights: attention score of each timesteps
        '''
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        embed = self.embedding(
            decoder_input
        )  # (batch_size, seq_len, embedding_dim), and seq_len must be 1
        embed = embed.permute(1, 0, 2)  # (seq_len,batch_size,embedding_dim)
        #embed = self.dropout(embed)

        output, (h, c) = self.lstm(
            embed, hidden
        )  #(seq_len,batch_size,hidden_size),tuple of (1,batch_size,hidden_size)

        x = self.linear(h)  #(1,batch_size,1)
        x = torch.sigmoid(x).squeeze(0)  #(batch_size,1)

        localmask = local_mask(x, self.d_size, seq_len) #(batch_size, seq_len)
        encoder_outputs = encoder_outputs.permute(
            1, 0, 2)  #(batch_size, seq_len, hidden_size)
        

        attn_scores = encoder_outputs.bmm(h.permute(
            1, 2, 0))  #(batch_size,seq_len,1)

        attn_scores.data.masked_fill_(localmask.unsqueeze(2), -float('inf'))
        attn_scores = attn_scores.squeeze(2)  #(batch_size,seq_len)
        
        attn_weights = torch.softmax(
            attn_scores, dim=1)  #(batch_size,seq_len) 
        attn_weights=attn_weights.view(batch_size, seq_len, 1) #(batch_size,seq_len,1)
        context = encoder_outputs.permute(0, 2, 1).bmm(attn_weights).squeeze(
            2)  #(batch_size,hidden_size)
        output = torch.cat([context, h.squeeze(0)],
                           dim=1)  #(batch_size,2*hidden_size)
        output = self.out(output)  #(batch_size, num_vocab)
        output = torch.reshape(
            output,
            (1, batch_size, self.num_vocab))  #(1, batch_size, num_vocab)
        return output, (h, c), attn_weights  #attn_weights will be used to visualize how attention works


class LocalAttentionEncoderDecoder(nn.Module):
    def __init__(self,
                 src_num_vocab,
                 tgt_num_vocab,
                 embedding_dim,
                 hidden_size,
                 src_embedding_matrix,
                 tgt_embedding_matrix,
                 dropout_p=0.2,
                 use_mask=True):
        super(LocalAttentionEncoderDecoder, self).__init__()
        self.encoder = BiEncoder(src_num_vocab, embedding_dim, hidden_size,
                                 src_embedding_matrix)
        self.decoder = LocalAttentionDecoder(
            tgt_num_vocab,
            embedding_dim,
            2 * hidden_size,
            maxlen,
            tgt_embedding_matrix,
            dropout_p=dropout_p)
        
        self.use_mask=use_mask

    def forward(self, src, tgt, lengths, dec_vocab, teacher_forcing_ratio):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        #encoder_outputs must be (seq_len,batch_size,hidden_size)

        tgt_length = tgt.size(1)  #tgt.shape(batch_size, seq_len)
        batch_size = tgt.size(0)
        
        mask=None
        if self.use_mask:
            src = pack_padded_sequence(src, lengths, batch_first=True)
            src, _ = pad_packed_sequence(src, batch_first=True)
            mask = torch.eq(src,0) #(batch_size, seq_len)

        outputs = []

        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id("<BOS>")] * batch_size,
            dtype=torch.long,
            device=device).unsqueeze(1)  #(batch_size,1)

        for i in range(tgt_length):
            is_teacher_forcing = True if np.random.random(
            ) < teacher_forcing_ratio else False            
            output, decoder_states, attn_weights = self.decoder(
                decoder_input, decoder_states,
                encoder_outputs, mask)  # (1,batch_size,vocab_size)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(output)
            if is_teacher_forcing:
                decoder_input = tgt[:, i].unsqueeze(1)
            else:
                #topi.detach()
                decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 2, 0)  #(batch_size,vocab_size,seq_len)
        return outputs
    
    def sample(self, src, lengths, tgt_length, dec_vocab):
        encoder_outputs, encoder_states = self.encoder(src, lengths)
        
        src = pack_padded_sequence(src, lengths, batch_first=True)
        src, _ = pad_packed_sequence(src, batch_first=True)
        mask = torch.ne(src,0) #(batch_size, seq_len)

        batch_size = src.size(0)
        decoder_states = encoder_states
        decoder_input = torch.tensor(
            [dec_vocab.word2id('<BOS>')] * batch_size,
            dtype=torch.long,
            device=device).unsqueeze(1)  #(batch_size, 1)

        outputs = []

        for i in range(tgt_length):
            output, decoder_states, attn_weights = self.decoder(
                decoder_input, decoder_states, encoder_outputs, mask)
            topv, topi = torch.max(output, 2)  # (1, batch_size)
            outputs.append(topi)  #greedy search
            decoder_input = topi.permute(1, 0)  #(batch_size, 1)

        outputs = torch.cat(
            outputs, dim=0).permute(1, 0)  #(batch_size,seq_len)
        return outputs    


# In[250]:

def remove_choice_number(text):
    '''文頭に選択肢番号がついている場合それを除く。
    前処理で使うだけなのでこのファイルでは呼び出さない。別のファイルに移したい。
    '''
    remove_list = [
        "^ア ", "^イ ", "^ウ ", "^エ ", "^オ ", "^1 ", "^2 ", "^3 ", "^4 ", "^5 "
    ]
    for i, word in enumerate(remove_list):
        text = re.sub(word, "", text)
    return text


def remove_symbol(text):
    '''
    入力されたテキストから句読点などの不要な記号をいくつか削除する。
    '''
    remove_list = [
        ',', '.', '-', '、', '，', '。', '\ufeff', '\u3000', '「', '」', '（', '）',
        '(', ')', '\n'
    ]
    for i, symbol in enumerate(remove_list):
        text = text.replace(symbol, '')
    return text


def add_bos_eos(text):
    '''
    文章の先頭に<BOS>、<EOS>を加える。文末の改行コードの都合で<EOS>の直前にはスペースを入れていない。
    '''
    return "<BOS> " + text + "<EOS>"


def replace_number(text):
    '''textの数値表現を<Number>トークンに置き換える
    textは分かち書きされていること
    '''
    new_text = ""
    for word in text.split(' '):
        if word.isnumeric():
            new_text += "<NUM> "
        elif word == "<EOS>":
            new_text += "<EOS>"
        else:
            new_text += word + " "
    return new_text


def isalpha(s):
    '''
    Args:
        s:string
    Returns:
        bool:sが半角英字から成るかどうか
    '''
    alphaReg = re.compile(r'^[a-zA-Z]+$')
    return alphaReg.match(s) is not None


def replace_alphabet(text):
    '''
    Args:
    text:分かち書きされた文。
    Return:
    textの数値表現をAに置き換える
    '''
    new_text = ""
    for word in text.split(' '):
        if isalpha(word):
            new_text += "A "
        elif word == "<EOS>":
            new_text += word
        else:
            new_text += word + " "
    return new_text


def local_mask(x, d_size, seq_len):
    center = seq_len * x
    size = x.size(0)
    a = torch.arange(seq_len * size).resize_(size, seq_len)
    a = a % seq_len
    a = a.float()
    a = a.cuda()
    #x is (batch_size, 1)
    #(batch_size, seq_len)のテンソルを作る。要素は[0,1,2...,seq_len-1]*batch_size
    #torch.gt, torch.ltを使ってmaskを作る
    assert a.size(0) == center.size(0), '{}, {}'.format(a.size(0), center.size(0))
    b = torch.lt(a, (center - d_size).float())
    c = torch.gt(a, (center + d_size).float())
    mask = b + c
    #mask = 1 - 100 * mask
    #mask = mask.float()

    return mask


# In[251]:

def train(src,
          tgt,
          lengths,
          model,
          optimizer,
          criterion,
          dec_vocab,
          is_train=True,
          teacher_forcing_ratio=0.8):
    '''一回のミニバッチ学習
    Args:
        src:入力文
        tgt:正解文
        model:EncoderDecoderモデル
        optimizer:torch.optim
        criterion:損失関数
        decoder_vocab:Decoder側のVocabクラス
        is_train:bool
        teacher_forcing_ration:teacher forcingを実行する確率
    Returns:
        loss: averaged loss of all tokens
    '''

    #minibatch学習のたびにteacher forcingするかどうかをサンプルしている、単語ごとにサンプルするのがScheduled Sampling
    is_teacher_forcing = True if np.random.random(
    ) < teacher_forcing_ratio else False

    y_pred = model(src, tgt, lengths, dec_vocab, is_teacher_forcing)

    loss = criterion(y_pred, tgt)

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# In[252]:

def trainIters(model,
               criterion,
               train_dataloader,
               valid_dataloader,
               decoder_vocab,
               epochs=epochs,
               batch_size=batch_size,
               print_every=1,
               plot_every=5,
               teacher_forcing_ratio=0.8):
    '''Encoder-Decoderモデルの学習
    
    '''

    #validation dataがないのでしっかりそれも書く
    optimizer = optim.Adam(model.parameters())

    plot_losses = []

    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        valid_loss = 0
        for batch_id, (batch_X, batch_Y,
                       X_lengths) in enumerate(train_dataloader):
            loss = train(
                batch_X,
                batch_Y,
                X_lengths,
                model,
                optimizer,
                criterion,
                decoder_vocab,
                is_train=True,
                teacher_forcing_ratio=teacher_forcing_ratio)
            train_loss += loss
            if batch_id % print_every == 0:
                elapsed_sec = time.time() - start
                elapsed_min = int(elapsed_sec / 60)
                elapsed_sec = elapsed_sec - 60 * elapsed_min
                print(
                    'Epoch:{} Batch:{}/{} Loss:{:.4f} Time:{}m{:.1f}s'.format(
                        epoch, batch_id,
                        int(train_dataloader.size /
                            train_dataloader.batch_size),
                        train_loss / (1 + batch_id), elapsed_min,
                        elapsed_sec),
                    end='\r')
        print()

        for batch_id, (batch_X, batch_Y,
                       X_lengths) in enumerate(valid_dataloader):
            loss = train(
                batch_X,
                batch_Y,
                X_lengths,
                model,
                optimizer,
                criterion,
                decoder_vocab,
                is_train=False,
                teacher_forcing_ratio=0)
            valid_loss += loss
            if batch_id % plot_every:
                plot_losses.append(loss)

        mean_valid_loss = valid_loss / (1 + batch_id)
        print('Epoch:{} Valid Loss:{:.4f}'.format(epoch, mean_valid_loss))

    return plot_losses


# In[253]:

#preprocess
#If you already have processed data, you can skip this cell. 
#過去12年分の宅建の過去問
takken = pd.read_csv(DATA_PATH+"takken.csv", encoding='utf-8')
mondaishu = pd.read_csv(DATA_PATH+"mondaishu.csv", encoding='utf-8')
nikken = pd.read_csv(DATA_PATH+"nikken.csv",encoding='utf-8')
legal_mind = pd.read_csv(DATA_PATH+"legal_mind.csv",encoding='utf-8')

#データをまとめる
takken = takken[["Question", "Choice"]]
ocr = pd.concat([mondaishu,nikken,legal_mind],axis=0,ignore_index=True)
ocr = ocr[["Wakati_Question","Wakati_Choice"]]
ocr.columns = ["Question","Choice"]

#データをMeCabで分かち書きして、不要な文字の除去、BOS,EOSの追加を行う
m = MeCab.Tagger("-Owakati")
takken = takken.applymap(remove_symbol)
ocr = ocr.applymap(remove_symbol)
takken = takken.applymap(m.parse)
takken = pd.concat([takken,ocr],axis=0,ignore_index=True)
takken = takken.applymap(remove_symbol)
takken = takken.applymap(add_bos_eos)
takken = takken.applymap(replace_number)
takken = takken.applymap(replace_alphabet)
print("data size is",len(takken))


# In[1]:

#instead of the aboce cell, only this will make sense
takken = pd.read_csv(DATA_PATH + "train.csv", encoding='utf-8')


# In[254]:

##長い系列は推論が難しいので一定程度長い文はデータから取り除く
q_maxlen = 150
c_maxlen = 110
maxlen = max(q_maxlen, c_maxlen)

takken = takken[takken["Question"].str.split(' ').apply(len) <= q_maxlen]
takken = takken[takken["Choice"].str.split(' ').apply(len) ""<= c_maxlen]
takken.reset_index(drop=True, inplace=True)
input_lengths = takken['Choice'].str.split().apply(len)
input_lengths = np.array(input_lengths) -1
print(input_lengths)


# In[255]:

#make dictionary
c_words = Vocab()
q_words = Vocab()
c_words.add("<NUM>")
q_words.add("<NUM>")
for i in range(len(takken)):
    for word in (takken.loc[i, "Question"]).split():
        q_words.add(word)
    for word in (takken.loc[i, "Choice"]).split():
        c_words.add(word)
        
with open('choice.vocab','wb') as f:
    pickle.dump(c_words,f)

with open('question.vocab','wb') as f:
    pickle.dump(q_words,f)


# In[256]:

num_encoder_tokens = c_words.size
num_decoder_tokens = q_words.size
print("vocabulary size in choices is", num_encoder_tokens)
print("vocabulary size in questions is", num_decoder_tokens)


# In[257]:

#Embedding層の初期値としてpre-trainさせたword2vec embeddingを用いる。
#単語辞書の中にはword2vecモデルに含まれない単語もあるので、そのembeddingは一様乱数で初期化する
word2vec = Word2Vec.load("../wiki_textbook/text_wiki_model")

word2vec_size = 200
encoder_embedding_matrix = np.random.uniform(
    low=-0.05, high=0.05, size=(num_encoder_tokens, word2vec_size))
decoder_embedding_matrix = np.random.uniform(
    low=-0.05, high=0.05, size=(num_decoder_tokens, word2vec_size))


# In[258]:

unknown_set = set()

for i, word in enumerate(c_words):
    try:
        encoder_embedding_matrix[i] = word2vec[word]
    except KeyError:
        if word not in unknown_set:
            unknown_set.add(word)
for i, word in enumerate(q_words):
    try:
        decoder_embedding_matrix[i] = word2vec[word]
    except KeyError:
        if word not in unknown_set:
            unknown_set.add(word)
            
encoder_embedding_matrix[0]=np.zeros((word2vec_size,))
decoder_embedding_matrix[0]=np.zeros((word2vec_size,))

encoder_embedding_matrix = encoder_embedding_matrix.astype('float32')
decoder_embedding_matrix = decoder_embedding_matrix.astype('float32')

unknown_set.remove("<NUM>")
unknown_set.remove("<UNK>")
unknown_set.remove("<PAD>")
unknown_set.remove("<BOS>")
unknown_set.remove("<EOS>")


# In[259]:

unknown_set


# In[260]:

#Vocab classに合わせてlistで管理したい
datasize = takken.shape[0]
choice = np.zeros((datasize, c_maxlen),dtype='int32')
question = np.zeros((datasize, q_maxlen),dtype='int32')


# In[261]:

for i in range(datasize):
    for j, word in enumerate(takken.loc[i, "Choice"].split(' ')):
        if word in unknown_set:
            word = "<UNK>"
        choice[i][j] = c_words.word2id(word)
    for j, word in enumerate(takken.loc[i, "Question"].split(' ')):
        if word in unknown_set:
            word = "<UNK>"
        question[i][j] = q_words.word2id(word)
        
choice = choice[:, :-1]
question = question[:, 1:]


# In[262]:

criterion = nn.CrossEntropyLoss(ignore_index=0) #not to include <PAD> in loss calculation


# In[263]:

model = LocalAttentionEncoderDecoder(c_words.size,q_words.size,200,latent_dim,encoder_embedding_matrix,decoder_embedding_matrix).to(device)

input_lengths = np.array(input_lengths)

train_choice, valid_choice, train_question, valid_question, train_input_lengths, valid_input_lengths = train_test_split(
choice,question,input_lengths)

train_dataloader = DataLoader(train_choice,train_question,train_input_lengths,batch_size=batch_size)
valid_dataloader = DataLoader(valid_choice,valid_question,valid_input_lengths,batch_size=batch_size)


# In[264]:

losses = trainIters(model,criterion,train_dataloader,valid_dataloader,q_words)


# In[265]:

plt.figure(figsize=(20,8))
plt.plot(losses)
plt.savefig('saved_model/local_attention_decoder4_loss.png')


# In[266]:

torch.save(model.state_dict(), 'saved_model/local_attention_decoder.model')


# In[267]:

with open('choice.vocab','rb') as f:
    c_words = pickle.load(f)
    
with open('question.vocab','rb') as f:
    q_words = pickle.load(f)


# In[268]:

iteration = 100 // batch_size + 1
with open('./outputs/c2q_outputs.csv', 'w') as f:
    for i in range(iteration):
        valid_dataloader.reset()
        batch_X, batch_Y, X_length = next(valid_dataloader)
        tgt_length = batch_Y.size(1)
        y_pred = model.sample(batch_X, X_length, tgt_length, q_words)
        X = batch_X.tolist()
        Y_true = batch_Y.tolist()
        Y_pred = y_pred.tolist()
        for x, y_true, y_pred in zip(X, Y_true, Y_pred):
            x = c_words.ids2seq(x)
            y_true = q_words.ids2seq(y_true)
            y_pred = q_words.ids2seq(y_pred)
            x = ' '.join(x)
            y_true = ' '.join(y_true)
            y_pred = ' '.join(y_pred)
            print("入力文")
            print(x)
            print("正解")
            print(y_true)
            print("推論")
            print(y_pred)
            f.write(x + ',' + y_true + ',' + y_pred + '\n')

