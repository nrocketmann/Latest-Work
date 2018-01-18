#convolutional neural network to classify if an image is of a cat or a dog
#uses data from kaggle cats and dogs challenge

from scipy import misc
import os
import numpy as np
import tensorflow as tf
import random

#### DATA PREP
def file_opener(file):
    return misc.imread(file)

basename = 'basename_here'
directory = '/catsanddogs/train/setforpractice/'

sess = tf.InteractiveSession()


# animals[0] is cats animals[1] is dogs
# list of numpy arrays

def dict_creator(dir):
    animals = [[], []]
    for filename in os.listdir(dir):
        new_item = file_opener(dir + filename)
        if 'cat' in filename:
            animals[0].append(new_item)
        elif 'dog' in filename:
            animals[1].append(new_item)
    return animals

#puts everything into a LIST not a dict, sorry for writing it that way
#data[0] is all of the cats in [375,499,3] numpy arrays
#data[1] is all of the dogs in [375,499,3] numpy arrays
data = dict_creator(directory)


# turns a numpy array into a big vector in case you want to try softmax
# none of that code is included here though

def simple_vectors(datas):
    new_image = []
    for row in datas:
        for column in row:
            for rgb in column:
                new_image.append(float(rgb))


#### FUNCTION DEFINITIONS FOR CONVNET

#creates randomized weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#makes it easy to add bias to future calculations
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#first convolutional layer, zero padding, step size 1 in all directions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#first pooling layer, 2x2 box stride size 1
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
#### DEFINE VARIABLES
#initialize weights for first conv layer, 5x5 filter, 3 input channels, 32 outputs
W_conv1 = weight_variable([5, 5, 3, 32])
#adds bias variable
b_conv1 = bias_variable([32])
#placeholders for images and y values
x_ = tf.placeholder(tf.float32, shape=[375, 499, 3])
x_image = tf.reshape(x_, [-1, 375, 499, 3])
y_ = tf.placeholder(tf.float32, shape=[1, 2])
#first conv layer relu function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#first pooline layer
h_pool1 = max_pool_2x2(h_conv1)
#initialize second conv and pool layers
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#actual calculations for second layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([94 * 125 * 64, 2048])
b_fc1 = bias_variable([2048])
#more calculations...
h_pool2_flat = tf.reshape(h_pool2, [-1, 94 * 125 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([2048, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#cross entropy defined
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#ADAM optimizer, learning rate
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
# cats are [1,0]
# dogs are [0,1]
#prep y arrays
y_cat = np.array([1,0])
y_dog = np.array([0,1])
accuracies = []
#20000 iterations
for i in range(20000):
    #alternate cat and dog
    if i % 2 == 0:
        x_feed = data[0][random.randint(0, 999)]
        y_feed = y_cat
    else:
        x_feed = data[1][random.randint(0, 999)]
        y_feed = y_dog
    #print accuracy every 3 iterations
    if i % 3 == 0:
        #feed images, make them fit into placeholder arrays
        train_accuracy = accuracy.eval(feed_dict={
            x_image: np.reshape(x_feed,[1,375,499,3]), y_: np.reshape(y_feed,[1,2]), keep_prob: 1.0})
        accuracies.append(train_accuracy)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x_image: np.reshape(x_feed,[1,375,499,3]), y_: np.reshape(y_feed,[1,2]), keep_prob: 0.5})


# CATS
# 375x499 very common, only use those
# 1285 of the first 10k images have those dimensions
# DOGS
# 1038 of those dimensions
