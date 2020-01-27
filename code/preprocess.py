import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import gensim
import re
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.functional import mse_loss 
import pickle
word_limit = 8
import matplotlib.pyplot as plt

train_path = "./LIAR-PLUS-master/dataset/train2.tsv"
test_path = "./LIAR-PLUS-master/dataset/test2.tsv"
valid_path = "./LIAR-PLUS-master/dataset/val2.tsv"
google_path = "./GoogleNews-vectors-negative300.bin"
save_path = "./data"

word2vec = gensim.models.KeyedVectors.load_word2vec_format(google_path, binary=True)

def clean_sentence(s):
    regex = re.compile('[^a-zA-Z]')
    new_sent = []
    for w in s.split(" "):
        w = regex.sub('', w)
        if(w == "didnt"):
            new_sent.append("did")
            new_sent.append("not")
        elif(w == "isnt"):
            new_sent.append("is")
            new_sent.append("not")
        elif(w == "wasnt"):
            new_sent.append("was")
            new_sent.append("not")
        elif(w == "doesnt"):
            new_sent.append("does")
            new_sent.append("not")
        elif(w == "hasnt"):
            new_sent.append("has")
            new_sent.append("not")
        elif(w == "bushs"):
            new_sent.append("bush")  
        else:    
            new_sent.append(w)
    s = " ".join(new_sent)
    return s.strip().lower()

def clean_subject(s):
    new_sub = []
    for w in s.split(","):
        sub = w.split("-")[0].lower()
        if(sub in word2vec):
            new_sub.append(sub)
    if(len(new_sub) == 0):
        return "<pad>"
    else:
        return " ".join(new_sub)
        
def clean_party(s):
    new_party = []
    for p in s.split("-"):
        p = p.lower()
        if(p in word2vec):
            new_party.append(p)
    if(len(new_party) == 0):
        return "<pad>"
    else:
        return " ".join(new_party)


def clean_df(df):
    df = df[df[15].map(lambda x: isinstance(x, str))]
    df = df[df[3].map(lambda x: isinstance(x, str))]
    df = df[df[4].map(lambda x: isinstance(x, str))]
    df[15] = df[15].map(lambda x : clean_sentence(x))
    df[3] = df[3].map(lambda x : clean_sentence(x))
    df[4] = df[4].map(lambda x : clean_subject(x))
    df[8] = df[8].map(lambda x : clean_party(x))
    cred = df[[9,10 ,11, 12,13]]
    df["cred"] = cred.values.tolist()
    return df

def get_data(tp):
    if(tp == "train"):
        path = train_path
    elif(tp == "valid"):
        path = valid_path
    elif(tp == "test"):
        path = test_path
       
    df  = pd.read_csv(path , sep = "\t" ,  header  = None)
    print("Orignal len " ,len(df))
    df = clean_df(df)
    print("len after cleaning " , len(df))
    return df

train_data = get_data("train")
test_data = get_data("test")
valid_data = get_data("valid")

sents = train_data[3].tolist() + valid_data[3].values.tolist() + test_data[3].values.tolist() + train_data[15].tolist() + valid_data[15].values.tolist() + test_data[15].values.tolist() + train_data[4].tolist() + valid_data[4].values.tolist() + test_data[4].values.tolist() + train_data[8].tolist() + valid_data[8].values.tolist() + test_data[8].values.tolist()

def build_dict(sents , word2vec):
    word2count = {}
    for s in sents:
        for w in s.split(" "):
            if(w not in word2count):
                word2count[w] = 1
            else:
                word2count[w] = word2count[w] + 1            
    word2index = {}
    index2word = {}
    word2index["<pad>"] = 0
    word2index["<bos>"] = 1
    word2index["<eos>"] = 2
    word2index["<unk>"] = 3
    index2word[0] = "<pad>"
    index2word[1] = "<bos>"
    index2word[2] = "<eos>"
    index2word[3] = "<unk>"
    index = 4
    embedding = []
    for i in range(index):
        embedding.append(np.random.normal(size = 300))
    count = 0
    for w in word2count:
        if(w in word2vec):
            word2index[w] = index
            index2word[index] = w
            embedding.append(word2vec[w]) 
            index = index + 1
        elif(word2count[w] > word_limit):
            count = count + 1
            word2index[w] = index
            index2word[index] = w
            embedding.append(np.random.normal(size = 300))
            index = index + 1
    print(count , "new words randomly init")
    return word2index , index2word , np.array(embedding)
print("Building word2index , index2word and embedding")
word2index , index2word  ,embedding = build_dict(sents , word2vec)

def save_dict(path , dct):
    with open(path , "wb") as handle:
        pickle.dump(dct , handle , protocol = pickle.HIGHEST_PROTOCOL)
def load_dict(path):
    with open(path, "rb") as handle:
        b = pickle.load(handle)
    return b

save_dict(os.path.join(save_path , "word2index.pkl") , word2index)
save_dict(os.path.join(save_path , "index2word.pkl") , index2word)

np.save(os.path.join(save_path , "embedding.npy") , embedding)