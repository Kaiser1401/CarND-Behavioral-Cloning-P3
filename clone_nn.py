# CarND behavioral cloning - nng

import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda
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

    # todo: load left and r and offset the angle!
    for line in lines:

        angle = 0
        try:
            angle = float(line[3])
        except:
            continue

        src = line[0]
        if brelocate:
            fn = src.split('/')[-1]
            src = sNewDir+fn
        img=cv2.imread(src)


        images.append(img)

        measurements.append(angle)

    return (np.array(images), np.array(measurements))

def get_nn():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) #normalize
    model.add(Flatten())
    model.add(Dense(1))
    return model



def train_nn(mod,X,y,valSplit=0.2,epochs=5, rate=0.001):
    mod.compile(loss='mse', optimizer=Adam(lr=rate))
    mod.fit(X,y,validation_split=valSplit,shuffle=True,epochs=epochs)
    return mod

def main():

    X,y = load_data(['./recordings/001/driving_log.csv','./recordings/002/driving_log.csv'])
    nn = get_nn()
    train_nn(nn,X,y)
    nn.save('model.h5')




if __name__ == '__main__':
    main()

