import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
import pickle
import collections as col
import os

os.chdir('/users/nameerhirschkind/machine-learning/WordVectors/')


def create_dict(size):
    tokens = [t.lower() for t in word_tokenize(open("corpus.txt").read())]
    d = {}
    for n, w in enumerate([item[0] for item in col.Counter(tokens).most_common(size)]):
        d[w] = n
    d['UNK'] = n+1
    rd = {n:w for w, n in d.items()}
    nt = []
    for token in tokens:
        if token not in d.keys():
            nt.append('UNK')
            continue
        nt.append(token)
    return d, rd, nt

def saveDicts(d, rd, tokens, batch_size, search_size, data_len):
    with open('dictionary.pickle', 'wb') as df:
        df.write(pickle.dumps(d))
    with open('reverse-dictionary.pickle','wb') as rdf:
        rdf.write(pickle.dumps(rd))
    with open("DataFile.pickle",'wb') as dtf:
        feed = {'inputs':[],'labels':[]}
        max_val = len(tokens)-1-search_size
        min_val = search_size
        for n in range(data_len):
            inp, label = np.zeros(shape=[batch_size]), np.zeros(shape=[batch_size,1])
            for i in range(batch_size):
                ind = np.random.randint(min_val,max_val)
                skip = np.random.choice(list(range(-search_size,0))+list(range(1,search_size+1)))+ind
                inp[i] = d[tokens[ind]]
                label[i][0] = d[tokens[skip]]
            feed['inputs'].append(inp)
            feed['labels'].append(label)
        dtf.write(pickle.dumps(feed))
    return True

def loadData(dict_size, batch_size, search_size,data_len):
    d, rd, nt = create_dict(dict_size)
    saveDicts(d,rd,nt, batch_size,search_size,data_len)

def saveEmbeddings(emb):
    f = open('embeddings.pickle','wb')
    f.write(pickle.dumps(np.squeeze(emb,0)))
    return True

#make new dataset
loadData(3000, 32, 4, 100000)


dict_size = 3001
batch_size = 32
skip_size = 4
data_len = 100000
embedding_size = 100
sampled = 1000
lrate = .005

with tf.Graph().as_default():
    embeddings = tf.Variable(tf.random_uniform([dict_size, embedding_size],-1.0,1.0))
    bias = tf.Variable(tf.zeros([dict_size]))
    weights = tf.Variable(tf.truncated_normal(shape=[dict_size,embedding_size],stddev=1.0/np.sqrt(embedding_size)))
    x = tf.placeholder(dtype=tf.int32,shape=batch_size,name="x")
    y = tf.placeholder(dtype=tf.int32,shape=[batch_size,1],name="y")
    embed = tf.nn.embedding_lookup(embeddings, x)
    loss = tf.nn.nce_loss(weights=weights,biases=bias,labels=y,inputs=embed,num_sampled=sampled,num_classes=dict_size)
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
    initializer = tf.global_variables_initializer()

    data = pickle.loads(open('DataFile.pickle','rb').read())
    trainX, trainY = data['inputs'], data['labels']
    sess = tf.Session()
    sess.run(initializer)
    total_loss = 0

    print("starting main loop...\n\n")
    for i in range(30000):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:trainX[i],y:trainY[i]})
        total_loss+=loss_val
        if i!=0 and i%1000==0:
            print('average loss at step {0}: {1}'.format(i,np.sum(total_loss)/1000))
            total_loss = 0
    final_embeddings = sess.run([embeddings])
    saveEmbeddings(final_embeddings)