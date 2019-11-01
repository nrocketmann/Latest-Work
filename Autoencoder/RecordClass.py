import cv2
import os
import shutil
import time
import pickle
import imageio
import numpy as np

#default record shape: (720, 1280, 3)

class Record:
    def __init__(self):
        pass

    def new_stream_dir(self,dump_name):
        try:
            os.mkdir('Recordings/'+dump_name)
        except FileExistsError:
            print('wiped directory ' + dump_name)
            shutil.rmtree('Recordings/' + dump_name)
            os.mkdir('Recordings/' + dump_name)

    def record(self,seconds,frame_gap,dump_name,size):
        self.new_stream_dir(dump_name)

        frames = []
        cap = cv2.VideoCapture(0)

        t = time.time()

        cnt = 0

        print('recording')
        while(True):
            if time.time() - t > seconds:
                break
            ret, frame = cap.read()
            square = cv2.resize(self.squarify(frame),(size,size))
            if cnt%frame_gap==0:
                frames.append(square)
            cnt+=1
            cv2.imshow('frame', square)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


        print('finished recording!')
        print('you recorded ' + str(len(frames)) + ' frames')
        frames = list(map(lambda x: np.flip(x,-1),frames))

        pickle.dump(frames,open('Recordings/' + dump_name + '/frames.pkl','wb'))
        pickle.dump({'frames':len(frames),'spacing':frame_gap,'seconds':seconds,
                     'real_fps':cnt/seconds, 'recorded_fps':len(frames)/seconds,'size':size},
                    open('Recordings/'+dump_name + '/metadata.pkl','wb'))
        print(frames[0].shape)

        return True

    def clear(self,name):
        try:
            shutil.rmtree('Recordings/' + name)
        except FileNotFoundError:
            print('could not find file ' + name)
            return False
        return True

    def playVid(self,dir):

        try:
            metadata = pickle.load(open('Recordings/'+dir+'/metadata.pkl','rb'))
            frames = pickle.load(open('Recordings/' + dir + '/frames.pkl','rb'))

        except FileNotFoundError:
            print('could not find directory')
            return False

        print('playing video from ' + dir + ': ' + str(metadata))
        cap = cv2.VideoCapture(0)
        for frame in frames:
            ret,f = cap.read()
            cv2.imshow('frame',np.flip(frame,-1))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1/metadata['recorded_fps'])

        return True


    def __str__(self):
        final = "Records:\n"
        for f in os.listdir('Recordings/'):
            final+=f
            final+=": "
            final+=str(pickle.load(open('Recordings/'+f+'/metadata.pkl','rb')))
            final+='\n'
        return final

    @staticmethod
    def all_recordings():
        return os.listdir('Recordings/')

    @staticmethod
    def squarify(im):
        return im[:,279:999,]

    @staticmethod
    def split_up(dirname):
        frames = pickle.load(open('Recordings/' + dirname + '/frames.pkl','rb'))
        try:
            os.mkdir('Recordings/' + dirname + '/images')
        except FileExistsError:
            print('images already exist')
            return False
        for i,frame in enumerate(frames):
            imageio.imwrite('Recordings/'+dirname+'/images/'+'image'+str(i)+'.jpg',frame)
        print(str(len(frames)) + ' frames saved')
        return True


def execute(time,frameskip,name,size):
    r = Record()
    r.record(time,frameskip,name,size)
    print(r)
    r.playVid(name)
    Record.split_up(name)

import argparse
import sys

if len(sys.argv)>2:
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--time", type=int, required=True,
        help="length of video in seconds")
    ap.add_argument("-fs", "--frameskip", type=int, required=True,
        help="number of frames to skip")
    ap.add_argument("-n", "--name", type=str, required=True,
        help="name for collection of files")
    ap.add_argument("-s","--size",type=int,required=True,
                    help="size of a side of the image (all are square)")

    args = vars(ap.parse_args())
    execute(args['time'],args['frameskip'],args['name'],args['size'])

else:
    print(Record.all_recordings())
