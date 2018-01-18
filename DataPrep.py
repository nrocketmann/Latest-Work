import pickle
import numpy as np
from nltk.tokenize import word_tokenize
import os

os.chdir('/users/nameerhirschkind/machine-learning/')

seq_len = 64
batch_size = 16
len_data = 150

d = pickle.loads(open('WordVectors/dictionary.pickle','rb').read())
tokens = [t.lower() for t in word_tokenize(open('WordVectors/corpus.txt').read())]
embeddings = pickle.loads(open('WordVectors/wordembeddings.pickle','rb').read())
embedding_size = len(embeddings[tokens[0]])
c = len(tokens)

nt = []
for token in tokens:
    if token not in d.keys():
        nt.append('UNK')
        continue
    nt.append(token)
tokens = nt

def gen_batch():
    Xbatch = []
    Ybatch = []
    for i in range(batch_size):
        ind = np.random.randint(0,c-seq_len-2)
        indy = ind+1
        wordsx = tokens[ind:ind+seq_len]
        wordsy = tokens[indy:indy+seq_len]
        embeddedx = np.array([embeddings[w] for w in wordsx]).reshape([seq_len,embedding_size])
        embeddedy = np.array([embeddings[w] for w in wordsy]).reshape([seq_len,embedding_size])
        Xbatch.append(embeddedx)
        Ybatch.append(embeddedy)
    return np.array(Xbatch).reshape([batch_size,seq_len,embedding_size]),np.array(Ybatch).reshape([batch_size,seq_len,embedding_size])

def createData(n):
    data = {'input':[],'label':[]}
    for i in range(len_data):
        b = gen_batch()
        data['input'].append(b[0])
        data['label'].append(b[1])
    f = open('LSTM/traindata/TrainData.pickle' + str(n),'wb')
    f.write(pickle.dumps(data))
    f.close()
    return True
def makeAllData(number):
    for i in range(number):
        createData(i)

makeAllData(40)