import os
import cv2
import tensorflow as tf
import numpy as np
import pickle
from RecordClass import Record
from matplotlib import pyplot as plt
import time
import copy


encoder = tf.keras.models.load_model('autoencoder/encoder.h5')

def processFrames(frames):
    encoded = []
    for frame in frames:
        encoded.append(encoder.predict(np.reshape((frame/255).astype(np.float32),[1,200,200,3])))
    return encoded

def process(name):
    frames = pickle.load(open('Recordings/' + name + '/frames.pkl','rb'))
    return processFrames(frames)


def distances_im(im,mem):
    dists = []
    encoded = encoder.predict(np.reshape((im/255).astype(np.float32),[1,200,200,3]))
    for vec in mem:
        dists.append(float(np.arccos(np.dot(vec,encoded.T)/(np.linalg.norm(vec)*np.linalg.norm(encoded)))))
    return dists
def distances(im,others):
    dists = []
    for other in others:
        dists.append(float(np.arccos(np.dot(other,im.T)/(np.linalg.norm(other)*np.linalg.norm(im)))))
    return dists

def takeImage():

    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    # Read picture. ret === True on success
    #wait for camera to calibrate
    time.sleep(1)
    ret, frame = video_capture.read()
    # Close device
    video_capture.release()
    return np.flip(cv2.resize(Record.squarify(frame),(200,200)),-1)

def test():
    im = takeImage()
    #im=pickle.load(open('Recordings/' + 'test1' + '/frames.pkl','rb'))[0]
    plt.imshow(im)
    plt.show()
    dists = distances(im,process('test1'))[1:]
    print(dists)
    print(np.mean(dists))
    print(min(dists))

def inVideo(name):
    im = takeImage()
    plt.imshow(im)
    plt.show()
    processed = process(name)
    cop = copy.copy(processed)
    internal_dists = []
    for i in range(len(processed)):
        rem = processed[i]
        cop.pop(i)
        internal_dists+=np.sort(distances(rem,cop))[:4].tolist()
        cop.insert(i,rem)
    int_mean, int_std = np.mean(internal_dists),np.std(internal_dists)
    ext_mean = np.mean(np.sort(distances_im(im,processed))[:4].tolist())
    if ext_mean-int_mean<=.75*int_std:
        return True,int_mean,int_std,ext_mean
    return False,int_mean,int_std,ext_mean

# test()
print(inVideo('test2'))
