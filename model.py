# CarND behavioral cloning - nng

import csv
import cv2
import numpy as np
import os
import math

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPool2D,Cropping2D,Dropout,AveragePooling2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_data(files,brelocate = False, sNewDir=''):
    lines = []
    for f in files:
        with open(f) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                if brelocate:
                    src=line[0]
                    fn = src.split('/')[-1]
                    src = sNewDir + fn
                    line[0]=src

                #check existance and angle is number
                try:
                    angle = float(line[3])
                except:
                    continue

                if os.path.isfile(line[0]):
                    lines.append(line)
    return lines



def load_images(csv_lines):
    images = []
    measurements = []

    for line in csv_lines:
        src = line[0]
        angle = float(line[3])
        img = cv2.imread(src)
        #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # quater pixels for faster learning :p! // uses average pooling now instead in the NN
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        measurements.append(angle)

    return ([images, measurements])


def load_data(files,brelocate = False, sNewDir=''):
    lines = read_data(files,brelocate,sNewDir)
    images, measurements = load_images(lines)
    return ([images, measurements])



def enhance(img,ang):
    # nothing yet
    new_img = []
    new_ang = []
    for i in range(len(ang)):
        new_ang.append(ang[i]*-1)
        new_img.append(cv2.flip(img[i],1))

    img.extend(new_img)
    ang.extend(new_ang)
    return [img,ang]

def numpyfy(a):
    return np.array(a)

def get_nn():
    model = Sequential()
    #model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(80,160,3))) #normalize
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))  # normalize

    model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid')) # try to do 'resizing' here instead of preprossesing to improve video :p

    model.add(Cropping2D(cropping=((36,13),(0,0)))) # crop to road


    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
    #model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))  # drop every second layer, for faster learning :p, network seems to be pretty complex anyway
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))

    #model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="relu"))

    model.add(Dropout(0.4)) # dropout for more robustnes

    model.add(Flatten())

    model.add(Dense(100))
    #model.add(Dense(50))

    model.add(Dropout(0.4)) # dropout for more robustnes

    model.add(Dense(10))

    model.add(Dense(1))
    return model

def data_generator(sample_lines,batch_size=128):
    count = len(sample_lines)
    while 1: # we need to start over after one epoch has 'used' all the data
        sample_lines = shuffle(sample_lines) # get a different order each time
        for start in range(0,count,batch_size):
            batch_lines = sample_lines[start:start+batch_size]
            X,y = load_images(batch_lines)
            yield [numpyfy(X),numpyfy(y)]


def train_nn(mod,X,y,valSplit=0.2,epochs=2, rate=0.001):
    mod.compile(loss='mse', optimizer=Adam(lr=rate))
    mod.fit(X,y,validation_split=valSplit,shuffle=True,epochs=epochs)
    return mod


def train_nn_with_generator(mod,lines,valSplit=0.2,epochs=2, rate=0.001, batch_size=128):
    lines = shuffle(lines)
    train_lines, val_lines = train_test_split(lines, test_size=valSplit)

    train_gen = data_generator(train_lines,batch_size=batch_size)
    val_gen = data_generator(val_lines,batch_size=batch_size)

    tr_steps = math.ceil(len(train_lines)/batch_size)
    vl_steps = math.ceil(len(val_lines) / batch_size)

    mod.compile(loss='mse', optimizer=Adam(lr=rate))
    #mod.fit_generator(train_gen, samples_per_epoch=len(train_lines), validation_data=val_gen, nb_val_samples=len(val_lines), nb_epoch=epochs,verbose=1)
    mod.fit_generator(train_gen, steps_per_epoch=tr_steps, validation_data=val_gen, validation_steps=vl_steps, epochs=epochs, verbose=1)
    return mod




def main():
    # old loader
    #X,y = load_data(['./recordings/001/driving_log.csv','./recordings/002/driving_log.csv','./recordings/003/driving_log.csv'])

    #new genarator approach
    # read csv lines first, only images are loaded by generator
    all_lines = read_data(['./recordings/001/driving_log.csv','./recordings/002/driving_log.csv','./recordings/003/driving_log.csv'])

    nn = get_nn()
    train_nn_with_generator(nn, all_lines)

    #X=numpyfy(X)
    #y = numpyfy(y)

    #train_nn(nn,X,y)
    nn.save('model3.h5')


if __name__ == '__main__':
    main()

