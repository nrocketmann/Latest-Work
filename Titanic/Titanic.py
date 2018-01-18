import pandas as pd
import tensorflow as tf
import numpy as np
import time
import random

df = pd.read_csv("titanic.csv",sep=",")
del df["Name"]
del df["PassengerId"]
labels = pd.Series(df["Survived"])
del df["Survived"]
df = df.drop("Ticket",axis=1)
df = df.drop('Cabin',axis=1)
df = df.fillna(value=0)
def oneHot(col,df):
    onehot = pd.get_dummies(df[col])
    df = df.drop(col,axis=1)
    df = df.join(onehot,lsuffix=col,rsuffix=col)
    return df
to_hot = ["Pclass",'Sex','Embarked',"SibSp","Parch"]
for x in to_hot:
    df = oneHot(x,df)
batch_size=100
dense = pd.DataFrame(df["Age"]).join(df["Fare"])
sparse = pd.DataFrame(df.drop("Age",axis=1).drop("Fare",axis=1))
def gen_batch(train=True):
    if train:
        ind=750
        a = 0
    else:
        ind = 890
        a = 750
    sparse_features = np.zeros(shape=[batch_size,23])
    dense_features = np.zeros(shape=[batch_size,2])
    y = []
    for i in range(batch_size):
        r = random.randint(a,ind)
        sparse_features[i] = np.array(list(sparse.iloc[r]))
        dense_features[i] = np.array(list(dense.iloc[r]))
        y.append(np.array(int(labels.iloc[r])))
    for i, r in enumerate(y):
        if r==1:
            y[i] = [1,0]
            continue
        y[i] = [0,1]

    return sparse_features,dense_features, np.array(y)

layer1_dense = 16
layer1_sparse = 32
layer2_sparse = 16
layer3_sparse = 16

with tf.Graph().as_default():
    sp = tf.placeholder(dtype=tf.float32,shape=[batch_size,23])
    dn = tf.nn.l2_normalize(tf.placeholder(dtype=tf.float32,shape=[batch_size,2]),dim=1)
    Y = tf.placeholder(dtype=tf.float32, shape=[batch_size,2])
    layer1db = tf.Variable(tf.zeros(shape=[layer1_dense]))
    layer1dw = tf.Variable(tf.random_normal(shape=[2,layer1_dense]))
    layer1sw = tf.Variable(tf.random_normal(shape=[23,layer1_sparse]))
    layer2sw = tf.Variable(tf.random_normal(shape=[layer1_sparse,layer2_sparse]))
    layer3sb = tf.Variable(tf.zeros(shape=[layer3_sparse]))
    layer3sw = tf.Variable(tf.random_uniform(shape=[layer2_sparse,layer3_sparse]))
    finw = tf.Variable(tf.random_normal(shape=[32,2]))
    l1dense = tf.nn.sigmoid(tf.matmul(dn, layer1dw)+layer1db)
    l1sparse = tf.nn.relu(tf.matmul(sp, layer1sw))
    l2sparse = tf.nn.relu(tf.matmul(l1sparse, layer2sw))
    l3sparse = tf.nn.sigmoid(tf.matmul(l2sparse, layer3sw)+layer3sb)
    lf = tf.concat([l3sparse,l1dense],axis=1)
    Y_ = tf.argmax(Y,axis=-1)
    pred = (tf.nn.sigmoid(tf.matmul(lf, finw)))
    full_pred = tf.argmax(pred,axis=-1)
    loss = tf.reduce_sum(tf.square(tf.squeeze(pred)-Y))
    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=tf.squeeze(tf.matmul(lf, finw)),dim=-1)
    optimizer1 = tf.train.AdamOptimizer(learning_rate=.1).minimize(loss)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)
    optimizer3 = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    tloss = 0
    for i in range(100000):
        b = gen_batch()
        if i<=2000:
            _, l = sess.run([optimizer1, loss],feed_dict={sp:b[0],dn:b[1],Y:b[2]})
        elif i<=7000:
            _, l = sess.run([optimizer2, loss], feed_dict={sp: b[0], dn: b[1], Y: b[2]})
        else:
            _, l = sess.run([optimizer3, loss], feed_dict={sp: b[0], dn: b[1], Y: b[2]})
        tloss+=l
        if i%1000==0 and i!=0:
            print("average loss at epoch " + str(i) + ": " + str(tloss/1000))
            tloss=0
        if i%1000==0 and i!=0:
            b = gen_batch(train=False)
            real, p = sess.run([Y_, full_pred], feed_dict={sp:b[0],dn:b[1],Y:b[2]})
            accuracy = 1-abs(np.sum(real-p))/batch_size
            print("accuracy on test data: " + str(accuracy))
