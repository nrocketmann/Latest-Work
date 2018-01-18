import numpy as np
import tensorflow as tf
import pickle
import os
import collections as col


load = False

#for testing
os.chdir('/users/nameerhirschkind/machine-learning/')
f = open('wordvectors/wordembeddings.pickle','rb')
data = pickle.loads(f.read())
os.chdir('/users/nameerhirschkind/machine-learning/lstm/')
batch_size = 16
seq_len = 64
data_size = 100
hidden_size = 512
layers = 2

def cell(dim):
    return tf.nn.rnn_cell.LSTMCell(dim,state_is_tuple=True)
def multicell(num,dim):
    return tf.nn.rnn_cell.MultiRNNCell([cell(dim) for _ in range(num)],state_is_tuple=True)

with tf.Graph().as_default():

    x = tf.placeholder(dtype=tf.float32, shape=[batch_size,seq_len,data_size])
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size,seq_len,data_size])
    cells = multicell(layers,hidden_size)
    cells.zero_state(batch_size,dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(cells,x,dtype=tf.float32)
    # dims of output are [batch,seq_len,output_size]

    output_weights = tf.get_variable(name='output-weights', initializer=tf.random_normal(shape=[hidden_size,data_size]))
    output_bias = tf.get_variable(name='output-bias',initializer=tf.zeros(shape=[data_size]),dtype=tf.float32)

    unstacked_output = tf.unstack(output,axis=1)
    logits = [tf.matmul(out,output_weights)+output_bias for out in unstacked_output]
    unstacked_labels = tf.unstack(y,axis=1)
    losses = [tf.reduce_sum(tf.square(pred-l)) for pred, l in zip(logits, unstacked_labels)]
    loss = tf.reduce_sum(losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    if load:
        saver.restore(sess, 'SavedVariables/model.ckpt')


    def dist(a, b):
        return np.sqrt(np.sum(np.square(b - a)))

    def get_closest(vec):
        close = {}
        for word, vector in data.items():
            close[word] = dist(vector, vec)
        topk = col.Counter(close).most_common()[:-2:-1][0][0]
        return topk

    def processPreds(p):
        allstuff = []
        for word in p:
            allstuff.append([])
            for batch in word:
                allstuff[-1].append(get_closest(batch))
        return np.array(allstuff).T

    def train(chunk):
        data = pickle.loads(open('traindata/traindata.pickle' + str(chunk),'rb').read())
        inp, lab = data['input'],data['label']
        total_loss = 0
        for _ in range(100):
            i = np.random.randint(0,150)
            _, loss_val = sess.run([optimizer, loss], feed_dict={x:inp[i], y:lab[i]})
            total_loss+=loss_val
        preds = sess.run([logits], feed_dict={x:inp[i]})
        return total_loss/100/batch_size/seq_len, preds

    def longTrain(num):
        for _ in range(num):
            n = np.random.randint(0,39)
            t = train(n)
            print(t[0])
            print(processPreds(t[1]))
    longTrain(10)
    #saver.save(sess, 'SavedVariables/model.ckpt')