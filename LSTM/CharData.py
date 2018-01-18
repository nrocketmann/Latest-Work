import pickle
import collections as col
import numpy as np
import os

os.chdir('/users/nameerhirschkind/machine-learning/LSTM/')
batch_size = 32
data = {}
corp = open('corpus.txt').read()
all_chars = []
for char in corp:
    all_chars.append(char.lower())
uniques = col.Counter(all_chars).keys()
for n, item in enumerate(uniques):
    data[item] = n

file = open('chardictionary.pickle','wb')
file.write(pickle.dumps(data))

num_batches = 5000
length = 64
corplen = len(corp)
corp = corp.lower()
all_batches = {'input':[],'output':[]}
for batch in range(num_batches):
    tempx, tempy = [],[]
    for i in range(batch_size):
        indx = np.random.randint(corplen-2-length)
        indy = indx+1
        allx = [data[c] for c in corp[indx:indx+length]]
        ally = [data[c] for c in corp[indy:indy+length]]
        tempx.append(allx)
        tempy.append(ally)
    all_batches['input'].append(np.array(tempx))
    all_batches['output'].append(np.array(tempy))
f = open('chardata.pickle','wb')
f.write(pickle.dumps(all_batches))