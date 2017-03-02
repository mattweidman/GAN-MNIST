import sys

import numpy as np
import scipy.misc

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

import mnist_parse

# parameters
n_train = 5000
n_valid = 1000
classes = 10
n_filters = 32
kernel_size = (3,3)
pool_size = (2,2)
batch_size = 128
epochs = 12

# load images
rawX = mnist_parse.getImages().astype('float')
def x_wrangle(x_in):
    x = x_in / 255
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    return x
wrangledX = x_wrangle(rawX)
trainX = wrangledX[:n_train]
validX = wrangledX[n_train:n_train+n_valid]
_, _, img_rows, img_cols = trainX.shape

# load labels
rawY = mnist_parse.getLabels()
trainY = np.zeros((n_train, classes))
trainY[np.arange(n_train), rawY[:n_train]] = 1
validY = np.zeros((n_valid, classes))
validY[np.arange(n_valid), rawY[n_train:n_train+n_valid]] = 1

#scipy.misc.imsave('im0.png', trainX[0,0,:,:])
#scipy.misc.imsave('im1.png', trainX[1,0,:,:])
#scipy.misc.imsave('im2.png', trainX[2,0,:,:])
#scipy.misc.imsave('im3.png', trainX[3,0,:,:])
#print(trainY[:4])
#sys.exit()

# create model
model = Sequential()

# 2 convolutional layers with relu activations, dropout at end
model.add(Convolution2D(n_filters, kernel_size[0], kernel_size[1],
    border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(n_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# fully connected layer, relu, dropout, fc, softmax
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
        metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=batch_size, nb_epoch=epochs,
        verbose=1, validation_data=(validX, validY))
score = model.evaluate(trainX, trainY)
print()
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(validX, validY)
print('valid score:', score[0])
print('valid accuracy:', score[1])
