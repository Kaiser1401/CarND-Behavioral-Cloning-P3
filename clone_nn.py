# CarND behavioral cloning - nng

import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPool2D,Cropping2D,Dropout
from keras.optimizers import Adam

def load_data(files,brelocate = False, sNewDir=''):
    lines = []
    for f in files:
        with open(f) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                lines.append(line)


    images = []
    measurements = []

    # todo: load left and r and offset the angle?
    for line in lines:

        angle = 0
        try:
            angle = float(line[3])
        except:
            continue

        src = line[0]

        try: # allow to just delete bad images in the directory
            if brelocate:
                fn = src.split('/')[-1]
                src = sNewDir+fn
            img=cv2.imread(src)

            img=cv2.resize(img,(0,0),fx=0.5, fy=0.5) # quater pixels for faster learning :p!

            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)          
        except:
            continue


        images.append(img)

        measurements.append(angle)


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
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(80,160,3))) #normalize

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

def train_nn(mod,X,y,valSplit=0.2,epochs=2, rate=0.001):
    mod.compile(loss='mse', optimizer=Adam(lr=rate))
    mod.fit(X,y,validation_split=valSplit,shuffle=True,epochs=epochs)
    return mod

def main():
    X,y = load_data(['./recordings/001/driving_log.csv','./recordings/002/driving_log.csv','./recordings/003/driving_log.csv'])
    #X,y = enhance(X,y) # nope
    nn = get_nn()

    X=numpyfy(X)
    y = numpyfy(y)

    train_nn(nn,X,y)
    nn.save('model2.h5')




if __name__ == '__main__':
    main()

