import numpy as np
import pandas as pd
import os
import pickle
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import random
import matplotlib.pyplot as plt 
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gensim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "./data"
word2index_path = "./data/word2index.pkl"
index2word_path = "./data/index2word.pkl"
embedding_path = "./data/embedding.npy"
train_path = "./LIAR-PLUS-master/dataset/train2.tsv"
test_path = "./LIAR-PLUS-master/dataset/test2.tsv"
valid_path = "./LIAR-PLUS-master/dataset/val2.tsv"
model_save_path = "./binary_model"
max_claim = 51
min_claim = 2
max_just = 261
min_claim = 4
batch_size = 64
lr = .0001
rep = 10154//batch_size
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def save_dict(path , dct):
    with open(path , "wb") as handle:
        pickle.dump(dct , handle , protocol = pickle.HIGHEST_PROTOCOL)
def load_dict(path):
    with open(path, "rb") as handle:
        b = pickle.load(handle)
    return b

embedding = np.load(embedding_path).astype(np.float32)
word2index = load_dict(word2index_path)
index2word = load_dict(index2word_path)
embedding = torch.tensor(embedding).to(device).double()

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
        if(sub in word2index):
            new_sub.append(sub)
    if(len(new_sub) == 0):
        return "<pad>"
    else:
        return " ".join(new_sub)
        
def clean_party(s):
    new_party = []
    for p in s.split("-"):
        p = p.lower()
        if(p in word2index):
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
    df[8] = df[8].map(lambda x : clean_subject(x))
    cred = df[[9,10 ,11, 12,13]]
    df["cred"] = cred.values.tolist()
    df = df[df[2].map(lambda x : x == "true" or x == "false")  ]
    return df

def get_data(tp):
    if(tp == "train"):
        path = train_path
    elif(tp == "valid"):
        path = valid_path
    elif(tp == "test"):
        path = test_path
       
    df  = pd.read_csv(path , sep = "\t" ,  header  = None)
    print("Orignal training points " ,len(df))
    df = clean_df(df)
    print("cleaned training points " , len(df))
    return df

train_data = get_data("train")
train_data = train_data[[3,4,8,"cred" , 15 , 2]]
train_data.columns = ["claim" , "subjects" ,"party", "cred" , "just" , "label"]

valid_data = get_data("valid")
valid_data = valid_data[[3,4,8,"cred" , 15 , 2]]
valid_data.columns = ["claim" , "subjects" ,"party", "cred" , "just" , "label"]

test_data = get_data("test")
test_data = test_data[[3,4,8,"cred" , 15 , 2]]
test_data.columns = ["claim" , "subjects" ,"party", "cred" , "just" , "label"]

labels = list(train_data["label"].unique())
labels

label_dict = {'false' :0 , 'true':1 }

def cut_sentence(s , tp):
    new_s = [word2index["<bos>"]]
    for w in s.split(" "):
        if(w in word2index):
            new_s.append(word2index[w])
    if(tp == "claim"):
        mx = max_claim
    elif(tp == "just"):
        mx = max_just
    if(len(new_s) > mx):
        new_s = new_s[: mx]    
    new_s.append(word2index["<eos>"])    
    return new_s

def process_cred(arr):
    s = sum(arr)
    if(s ==0 ):
        return arr
    else:
        return [x/s for x in arr]

def process_data(df):
    df["claim"] = df["claim"].map(lambda x : cut_sentence(x , "claim"))
    df["just"] = df["just"].map(lambda x : cut_sentence(x , "just"))
    df["label"] = df["label"].map(lambda x : label_dict[x])
    df["cred"] = df["cred"].map(lambda x : process_cred(x))
    df["party"] = df["party"].map(lambda x : word2index[x])
    return df

train_df = process_data(train_data)
valid_df = process_data(valid_data)
test_df  = process_data(test_data)

just = train_df["just"].values.tolist()
claim = train_df["claim"].values.tolist()

def get_batch(df , batch_size):
    batch = df.sample(batch_size)
    claim = batch["claim"].values.tolist()
    just = batch["just"].values.tolist()
    party = torch.tensor(batch["party"].values.tolist())
    claim = list(map(lambda x : torch.tensor(x) , claim)) 
    claim = pad_sequence(claim , batch_first = True)
    if(claim.size(1) < 20):
        pad = torch.zeros(claim.size(0) , 20 - claim.size(1)).long()
        claim = torch.cat((claim , pad) , dim = 1)
    just = list(map(lambda x : torch.tensor(x) , just)) 
    just = pad_sequence(just , batch_first = True)
    label = torch.tensor(batch["label"].values)
    cred  = torch.tensor(batch["cred"].values.tolist())
    return claim.to(device) , just.to(device) , label.to(device) , cred.to(device).double() , party.to(device)
    

def get_test(df):
    batch = df
    claim = batch["claim"].values.tolist()
    just = batch["just"].values.tolist()
    party = torch.tensor(batch["party"].values.tolist())
    claim = list(map(lambda x : torch.tensor(x) , claim)) 
    claim = pad_sequence(claim , batch_first = True)
    if(claim.size(1) < 20):
        pad = torch.zeros(claim.size(0) , 20 - claim.size(1)).long()
        claim = torch.cat((claim , pad) , dim = 1)
    just = list(map(lambda x : torch.tensor(x) , just)) 
    just = pad_sequence(just , batch_first = True)
    label = torch.tensor(batch["label"].values)
    cred  = torch.tensor(batch["cred"].values.tolist())
    return claim.to(device) , just.to(device) , label.to(device) , cred.to(device).double() , party.to(device)

class Detector(nn.Module):
    def __init__(self , word2index = word2index , embedding = embedding , embedding_dim = 300 , dim_hidden = 128  , dim_output = 2 , dim_linear1 = 100, dim_linear2 = 30 , dim_cred = 30 , dim_party = 30):
        super().__init__()
        self.word2index = word2index
        self.embedding_dim = embedding_dim
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_linear1 = dim_linear1
        self.dim_linear2 = dim_linear2
        self.dim_cred = dim_cred
        self.dim_party = dim_party
        self.embedding = nn.Embedding(len(word2index), self.embedding_dim)
        self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.cred_vector =  nn.Linear(5 , self.dim_cred)
        self.conv1 = nn.Conv2d(1, 128, (8, self.embedding_dim ))
        self.conv2 = nn.Conv2d(1, 128, (4,self.embedding_dim ))
        self.conv3 = nn.Conv2d(1, 128, (3,self.embedding_dim ))
        self.party_vector = nn.Linear(self.embedding_dim , self.dim_party)
        self.cnn_vector =  nn.Linear(128*3 , self.dim_hidden)
        self.lstm1 = nn.LSTM(self.embedding_dim , self.dim_hidden , batch_first = True , bidirectional = True)
        self.lstm2 = nn.LSTM(self.embedding_dim , self.dim_hidden , batch_first = True , bidirectional = True)
        
        stacked_dim = self.dim_hidden*5 + self.dim_cred + self.dim_party
        
        self.stacked_vector = nn.Linear(stacked_dim , self.dim_linear1)
        self.vector_vector = nn.Linear(self.dim_linear1 , self.dim_linear2)
        self.vector_output = nn.Linear(self.dim_linear2 , self.dim_output)
        
        self.vector_output = nn.Linear(self.dim_linear2 , self.dim_output)
    def forward(self ,claim , just , cred , party ):
        emb_claim = self.embedding(claim)
        emb_just = self.embedding(just)
        emb_party = self.embedding(party)
        
        cnn_in = emb_claim.unsqueeze(1)
        cnn1 = self.conv1(cnn_in).squeeze()
        cnn1 = cnn1.mean(dim = 2)
        
        cnn2 = self.conv2(cnn_in).squeeze()
        cnn2 = cnn2.mean(dim = 2)
        
        cnn3 = self.conv3(cnn_in).squeeze()
        cnn3 = cnn3.mean(dim = 2)
        
        cnn_out = torch.cat((cnn1 , cnn2 , cnn3) , dim = 1)
        cnn_out = self.cnn_vector(cnn_out)
        
        cred = self.cred_vector(cred)
        party = self.party_vector(emb_party)
        claim , _ = self.lstm1(emb_claim)
        just, _ = self.lstm2(emb_just)
        claim = claim[: ,-1, :]
        just = just[: ,-1, :]
        stacked = torch.cat((claim ,cnn_out,  cred ,party ,  just ) , dim = 1)
        out = self.stacked_vector(stacked)

        out = self.vector_vector(out)

        out = self.vector_output(out)
        return out

model = Detector().double().to(device)

params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(params, lr)
crit_video = torch.nn.CrossEntropyLoss()
def loss1(out , targets):
    targets = targets.contiguous().view(-1)
    return crit_video(out , targets)

def calculate_accuracy(df , model):
    T = get_test(df)
    
    with torch.no_grad():
        pred = model(T[0] ,T[1], T[3] , T[4])
    _ , val = torch.max(pred , dim = 1)
    val = val.to("cpu").numpy()
    label = T[2].to("cpu").numpy()
    f1 = f1_score(label, val, average="macro")
    precision = precision_score(label, val, average="macro")
    recall = recall_score(label, val, average="macro")    
    accu = accuracy_score(label , val )
    return f1 , precision , recall , accu

def train(model = model ,data = train_df , epochs = 20 , rep = rep , batch_size = batch_size):
    f1_arr = []
    loss_arr = []
    accu_arr = []
    best_model = -1 
    best_f1 = -1
    
    for epoch in range(epochs):
        print("epoch - " , epoch+1)
        total_loss = 0
        for _ in range(rep):
            b = get_batch(data , batch_size)
            pred = model(b[0] , b[1], b[3] , b[4])
            optimizer.zero_grad()
            loss = loss1(pred , b[2])
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
     #   print(total_loss/rep)
        loss_arr.append(total_loss/rep)
        torch.save(model.state_dict(), os.path.join(model_save_path , "model" + str(epoch + 1) + ".pt"))
        f1 , pr , r , accu = calculate_accuracy(valid_df , model)
        f1_arr.append(f1)
        accu_arr.append(accu)
        if(f1 > best_f1):
            best_model = epoch +1
            best_f1 = f1
  #      print("validation f1 accuracy" , f1 , accu)
    return best_model , loss_arr ,accu_arr , best_f1

best_model , loss_arr , accu_arr , best_f1 = train()

print("finished training")
print("Retriving best model.....")
print("best_model at epoch" ,best_model)

final_model =  Detector().to(device).double()
final_model.load_state_dict(torch.load(os.path.join(model_save_path , "model" + str(best_model) + ".pt")  ))
test_f1 , test_accu , _ , _ =  calculate_accuracy(test_df , final_model)

print("Results on test set")
print("Accuracy" , test_accu)
print("F1 score" , test_f1)