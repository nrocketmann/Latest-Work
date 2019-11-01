import tensorflow as tf
import pickle
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
import numpy as np
import time
#tf.enable_eager_execution()
BATCH = 32

def imshow(im):
    plt.imshow(im[0, :, :,])
    plt.show()

allimages = []
for i in range(1,21):
    allimages+=pickle.load(open('images/ImageFile' + str(i) + '.0','rb'))
num_images = len(allimages)
allimages = [im.astype(np.float32) for im in allimages]
allimages = [tf.convert_to_tensor(image) for image in allimages]
dataset = tf.data.Dataset.from_tensor_slices(allimages).shuffle(5000).batch(BATCH)


#model building

#generator
#layer1
generator = Sequential()
generator.add(layers.Dense(8*8*3*256,input_shape=[100]))
generator.add(layers.LeakyReLU())
generator.add(layers.BatchNormalization())
generator.add(layers.Reshape([8,8,3*256]))

#layer 2
generator.add(layers.Conv2DTranspose(128*3,[5,5],padding='same',use_bias=False,strides=[1,1]))
generator.add(layers.LeakyReLU())
generator.add(layers.BatchNormalization()) #returns 8x8xchan

#layer3
generator.add(layers.Conv2DTranspose(32*3,[5,5],padding='same',use_bias=False,strides=[2,2]))
generator.add(layers.LeakyReLU())
generator.add(layers.BatchNormalization()) #returns 16x16xchan

#layer4
generator.add(layers.Conv2DTranspose(8*3,[7,7],padding='same',use_bias=False,strides=[2,2]))
generator.add(layers.LeakyReLU())
generator.add(layers.BatchNormalization()) #returns 32x32xchan

#layer5
generator.add(layers.Conv2DTranspose(3,[8,8],padding='same',use_bias=False,strides=[2,2]))
generator.add(layers.Activation('sigmoid')) #returns 64x64x3


#discriminator
#layer1
discriminator = Sequential()
discriminator.add(layers.Conv2D(8*3,[8,8],padding='same',strides=[2,2],input_shape=[64,64,3]))
discriminator.add(layers.Dropout(.2))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.BatchNormalization())

#layer2
discriminator.add(layers.Conv2D(32*3,[7,7],padding='same',strides=[2,2]))
discriminator.add(layers.Dropout(.2))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.BatchNormalization())

#layer3
discriminator.add(layers.Conv2D(128*3,[5,5],padding='same',strides=[2,2]))
discriminator.add(layers.Dropout(.1))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.BatchNormalization())

#layer4
discriminator.add(layers.Conv2D(256*3,[5,5],padding='same',strides=[1,1]))
discriminator.add(layers.Activation('tanh'))
discriminator.add(layers.BatchNormalization())

#layer5
discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(64,activation='tanh'))
discriminator.add(layers.Dense(1,activation='sigmoid'))


#save model
# checkpoint_path = 'checkpoints/cp.ckpt'
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

def genImage():
    noise = tf.random.normal([1,100],dtype=tf.float32)
    im = generator(noise,training=False)
    return im

# gentest = genImage()
# pred = discriminator(gentest,training=False)
# imshow(gentest)

# print(pred)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_preds,fake_preds):
#     real_loss = cross_entropy(tf.zeros_like(real_preds),real_preds)
#     fake_loss = cross_entropy(tf.ones_like(fake_preds),fake_preds)
      real_loss = tf.reduce_sum(-tf.math.log(tf.ones_like(real_preds)-real_preds))
      fake_loss=tf.reduce_sum(-tf.math.log(fake_preds))
      return real_loss+fake_loss

def generator_loss(outputs):
    #return cross_entropy(tf.zeros_like(outputs),outputs)
    return tf.reduce_sum(-tf.math.log(tf.ones_like(outputs)-outputs))

generator_optimizer = tf.keras.optimizers.Adam(.001)
discriminator_optimizer = tf.keras.optimizers.Adam(.0001)

def genDogs(num):
    inds = np.random.randint(0,num_images,num)
    return np.array([allimages[ind] for ind in inds])

test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
def accuracy(real_ims):
    noise = tf.random.normal([BATCH,100],dtype=tf.float32)
    fake_ims = generator(noise,training=True)
    fake_preds = discriminator(fake_ims,training=True)
    real_preds = discriminator(real_ims,training=True)
    acc1 = np.sum(tf.map_fn(lambda x: tf.math.round(x),tf.squeeze(fake_preds)))/32
    acc2 = np.sum(tf.map_fn(lambda x: 1-tf.math.round(x),tf.squeeze(real_preds)))/32
    #acc1 = test_accuracy(tf.ones_like(fake_preds),fake_preds)
    #acc2 = test_accuracy(tf.zeros_like(real_preds),real_preds)
#     print(real_preds)
#     print(fake_preds)
    return acc1,acc2 #fake accuracy, real accuracy
@tf.function
def iteration(real_ims):
    #first generate inputs
    noise = tf.random.normal([BATCH,100],dtype=tf.float32)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        fake_ims = generator(noise,training=True)

        fake_preds = discriminator(fake_ims,training=True)
        real_preds = discriminator(real_ims,training=True)

        disc_loss = discriminator_loss(real_preds,fake_preds)
        gen_loss = generator_loss(fake_preds)

        gen_grad = gen_tape.gradient(gen_loss,generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gen_grad,generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))
    return np.sum(disc_loss),np.sum(gen_loss),tf.reduce_sum(gen_grad[-1]), tf.reduce_sum(disc_grad[-1])

EPOCHS = 20
N = 50
for epoch in range(EPOCHS):
    gls = 0
    dls = 0
    grt = 0
    drt = 0
    print('new epoch')
    for i, batch in enumerate(dataset):
        dl,gl,gr,dr = iteration(batch)
        gls+=gl
        dls+=dl
        grt +=gr
        drt+=dr
        if (i+1)%N==0:
            print('epoch {0}, iteration {1}'.format(epoch,i+1))
            print('average discriminator loss: ' + str(dls/N))
            print('average generator loss: ' + str(gls/N))
            print('average generator gradient size: ' + str(grt/N))
            print('average discriminator gradient size: ' + str(drt/N))
            fake_ac,real_ac = accuracy(batch)
            print('real accuracy: {0}, fake accuracy: {1}'.format(real_ac,fake_ac))
            #print(discriminator(batch))
            imshow(genImage())
            plt.imshow(batch[0])
            plt.show()
            gls = 0
            dls = 0
            grt = 0
    generator.save('checkpoints/generator.h5')
    discriminator.save('checkpoints/discriminator.h5')
