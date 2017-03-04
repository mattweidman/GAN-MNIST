# using tutorial from http://www.kdnuggets.com/2016/07/mnist-generative-adversarial-model-keras.html

import numpy as np
import scipy.misc

import keras.layers as kl
import keras.layers.convolutional as klconv
import keras.layers.core as klc
import keras.layers.normalization as kln

import keras.models as km

import keras.optimizers as ko

import mnist_parse

n_train = 10
classes = 10
batch_size = 100
noise_size = 100

# load images
rawX = mnist_parse.getImages().astype('float')
def x_wrangle(x_in):
    x = x_in / 255
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    return x
wrangledX = x_wrangle(rawX)
trainX = wrangledX[:n_train]
_, _, img_rows, img_cols = trainX.shape

# noise input
noise = np.random.uniform(0, 1, size=[n_train, noise_size])

# generator
n_channels = 200
l_width = img_rows / 2
g_input = kl.Input(shape=[noise_size])
H = klc.Dense(n_channels*l_width*l_width, init='glorot_normal')(g_input)
H = kln.BatchNormalization(mode=2)(H)
H = klc.Activation('relu')(H)
H = klc.Reshape([n_channels, l_width, l_width])(H)
H = klconv.UpSampling2D(size=(2,2))(H)
H = klconv.Convolution2D(n_channels/2, 3, 3, border_mode='same',
        init='glorot_uniform')(H)
H = kln.BatchNormalization(mode=2)(H)
H = klc.Activation('relu')(H)
H = klconv.Convolution2D(n_channels/4, 3, 3, border_mode='same',
        init='glorot_uniform')(H)
H = kln.BatchNormalization(mode=2)(H)
H = klc.Activation('relu')(H)
H = klconv.Convolution2D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
g_V = klc.Activation('sigmoid')(H)
generator = km.Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=ko.Adam(lr=1e-4))

generated_images = generator.predict(noise)

def save_images(predict, img_name):
    imgs = predict.reshape(n_train, img_rows, img_cols)
    for i in range(len(imgs)):
        scipy.misc.imsave(img_name + str(i) + ".png", imgs[i])

save_images(generated_images, "imgs/mnist")
