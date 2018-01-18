import numpy as np
import tensorflow as tf
import pickle
import os
import collections as col


load = False
#TODO: one hot x and y, use softmax loss function, change test function for one-hot
os.chdir('/users/nameerhirschkind/machine-learning/lstm/')
f = open('chardictionary.pickle','rb')
data = pickle.loads(f.read())
data_size = max(data.values())+1
batch_size = 32
seq_len = 64
hidden_size = 512
layers = 2

def cell(dim):
    return tf.nn.rnn_cell.LSTMCell(dim,state_is_tuple=True)
def multicell(num,dim):
    return tf.nn.rnn_cell.MultiRNNCell([cell(dim) for _ in range(num)],state_is_tuple=True)

with tf.Graph().as_default():

    x_ = tf.reshape(tf.placeholder(dtype=tf.int32, shape=[batch_size,seq_len]),shape=[batch_size,seq_len,1])
    y_ = tf.reshape(tf.placeholder(dtype=tf.int32, shape=[batch_size,seq_len]),shape=[batch_size,seq_len,1])
    x = tf.reshape(tf.one_hot(x_,data_size,axis=-1,name='x-input'),[batch_size,seq_len,data_size])
    y = tf.reshape(tf.one_hot(y_,depth=data_size,name='y-label',axis=-1),[batch_size,seq_len,data_size])
    cells = multicell(layers,hidden_size)
    cells.zero_state(batch_size,dtype=tf.float32)
    output, state = tf.nn.dynamic_rnn(cells,x,dtype=tf.float32)
    # dims of output are [batch,seq_len,output_size]

    output_weights = tf.get_variable(name='output-weights', initializer=tf.random_normal(shape=[hidden_size,data_size]))
    output_bias = tf.get_variable(name='output-bias',initializer=tf.zeros(shape=[data_size]),dtype=tf.float32)

    unstacked_output = tf.unstack(output,axis=1)
    logits = [tf.matmul(out,output_weights)+output_bias for out in unstacked_output]
    unstacked_labels = tf.unstack(y,axis=1)
    losses = [tf.nn.softmax_cross_entropy_with_logits(labels=l, logits=pred) for pred, l in zip(logits, unstacked_labels)]
    loss = tf.reduce_sum(losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    if load:
        saver.restore(sess, 'SavedVariables/model.ckpt')



    def train(iter):
        traindata = pickle.loads(open('chardata.pickle','rb').read())
        inp, lab = traindata['input'],traindata['output']
        total_loss = 0
        for z in range(iter):
            i = np.random.randint(0,5000)
            _, loss_val = sess.run([optimizer, loss], feed_dict={x_:np.reshape(inp[i],newshape=[batch_size,seq_len,1]), y_:np.reshape(lab[i],newshape=[batch_size,seq_len,1])})
            total_loss += loss_val
            if z%100==0 and z!=0:
                print(total_loss/100)
                total_loss = 0
        #preds = sess.run([logits], feed_dict={x:inp[i]})
        return True
    train(100)
    saver.save(sess, 'SavedVariables/model.ckpt')