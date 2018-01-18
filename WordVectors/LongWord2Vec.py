import tensorflow as tf
import csv
from random import randint as rd
import numpy as np
import sys
import json
print('loading data...')
with open('/users/nameerhirschkind/nlp/nlp/vsize.txt') as f:
    vocabulary_size = int(f.readline()) + 1
    f.close()
with open('/users/nameerhirschkind/nlp/nlp/dict.json') as jfile:
    data = json.load(jfile)
text = []
with open('/users/nameerhirschkind/nlp/nlp/practice1.txt') as textFile:
    for line in textFile.readlines():
        sub = line.split(' ')
        for w in sub:
            text.append(w)
print('finished')
def saveVectors(vectors, d):
    pass
lrate = .1
print('initializing...')
vector_size = 150
batchSize = 147
x = tf.placeholder(dtype=tf.int32, shape=[batchSize], name='x_input')
y = tf.placeholder(dtype=tf.int32, shape=[batchSize,1], name='y_output')
W1 = tf.Variable(tf.random_uniform([vocabulary_size, vector_size], -1.0,1.0))
W2 = tf.Variable(tf.truncated_normal(shape=[vocabulary_size,vector_size]))
b2 = tf.zeros(shape=[vocabulary_size])
embed = tf.nn.embedding_lookup(W1,x)
cross_entropy = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=W2,biases=b2,labels=y,inputs=embed,num_sampled=5,num_classes=vocabulary_size))
optimizer = tf.train.AdagradOptimizer(learning_rate=lrate).minimize(cross_entropy)


validation_set = []
for i in range(16):
    validation_set.append(text[rd(0,len(text)-1)])
vsetnums = {}
vset = []
for t in validation_set:
    vsetnums[t] = data[t]
    vset.append(data[t])

valid_dataset = tf.constant(np.array(vset),dtype=tf.int32)
#cosine distance
norm = tf.sqrt(tf.reduce_sum(tf.square(W1), 1, keep_dims=True))
normalized_embeddings = W1 / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

def euc_dist(str1, str2):
    worda = tf.nn.embedding_lookup(W1,data[str1]).eval()
    wordb = tf.nn.embedding_lookup(W1,data[str2]).eval()
    dist = np.sum(worda-wordb)**2
    return dist


def findByKey(item, data):
    for k, v in data.items():
        if item==v:
            return k

def generate_batch(window_size, allWords, numExamples, d):
    total = len(allWords)
    xbatch = []
    ybatch = []
    stuff = int(batchSize/numExamples)
    for z in range(stuff):
        r = rd(0,total-1)
        for i in range(numExamples):
            xbatch.append(d[allWords[r]])
            r2 = rd(-window_size,-1)
            r1 = rd(1, window_size)
            chooser = rd(0,1)
            if chooser==0:
                ybatch.append(d[allWords[r1]])
            else:
                ybatch.append(d[allWords[r2]])
    return xbatch, ybatch
numExamples = 3
epochs = 1000000
window_size = 4
init = tf.global_variables_initializer()
def euclidean_distance(item, item2):
    return np.sqrt((np.array(item2)-np.array(item))**2)
def makeVector(num):
    sub = [0 for i in range(vocabulary_size)]
    sub[num] = 1
    return sub
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
print('finished')
with tf.Session() as sess:
    sess.run(init)
    costTotal = 0
    for i in range(epochs):
        g = generate_batch(window_size,text,numExamples,data)
        inp = g[0]
        y_ = np.array([g[1]]).T
        _, cost, Wvectors = sess.run([optimizer, cross_entropy, W1], feed_dict={x: inp, y: y_})
        costTotal += cost
        if i % 200 == 0 and i!=0:
            score = costTotal*5 / (batchSize)

            print(score)
            costTotal = 0
        if i%1000==0 and i!=0:
            for zx in range(4):
                worda = input('first word')
                wordb = input('second word')
                try:
                    stuff = euc_dist(worda,wordb)
                except KeyError:
                    continue
                print('distance between words: ' + str(stuff))

        if i%1000 ==0 and i!=0:
            sim = similarity.eval()
            for i in range(16):
                valid_word = validation_set[i]
                top_k = 16 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = findByKey(nearest[k], data)
                    log = '%s %s,' % (log, close_word)
                print(log)

    sess.close()
