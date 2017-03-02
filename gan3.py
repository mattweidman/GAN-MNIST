import numpy as np
import scipy.misc
import theano
import theano.tensor as T
import keras

import mnist_parse

N = 16

rawX = mnist_parse.getImages()
def x_wrangle(x_in):
    x = x_in / 255
    return x
trainX = x_wrangle(rawX)
feats_D = trainX.shape[1]


