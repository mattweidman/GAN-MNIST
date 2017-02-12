import numpy as np
import theano
import theano.tensor as T

import mnist_parse

# parameters
N = 10000
valN = 10000
lr = 3
epochs = 1000

# import data
rawX = mnist_parse.getImages()[:N]
rawY = mnist_parse.getLabels()[:N]
rawValX = mnist_parse.getImages()[N:N+valN]
rawValY = mnist_parse.getLabels()[N:N+valN]

# wrangle data
def x_wrangle(x_in):
    x = x_in.reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2])
    x = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)
    x -= x.mean(axis=1)[:, np.newaxis]
    x /= x.std(axis=1)[:, np.newaxis]
    return x
trainX = x_wrangle(rawX)
valX = x_wrangle(rawValX)
feats = trainX.shape[1]

classes = 10
trainY = np.zeros((N, classes))
trainY[np.arange(N), rawY] = 1
valY = np.zeros((valN, classes))
valY[np.arange(valN), rawValY] = 1

# initialize variables
x = T.dmatrix("x")
y = T.dmatrix("y")
r = 4 * np.sqrt(6 / (feats + classes))
w = theano.shared(np.random.uniform(-r, r, (feats, classes)), name="w")

# equations
h = T.nnet.sigmoid(T.dot(x,w))
p = T.nnet.softmax(h)
loss = (-y*T.log(p)).mean()
gw = T.grad(loss, w)
accuracy = T.eq(T.argmax(h, axis=1), T.argmax(y, axis=1)).mean()

# compile
train = theano.function(inputs=[x,y], outputs=[loss], updates=[(w, w-lr*gw)])
predict = theano.function(inputs=[x], outputs=[p])
evaluate = theano.function(inputs=[x,y], outputs=[accuracy])

# train
for i in range(epochs):
    loss = train(trainX, trainY)
    if (i % 100 == 0):
        print(loss[0])
acc = evaluate(trainX, trainY)[0]
print("training accuracy: " + str(acc))
vacc = evaluate(valX, valY)[0]
print("validation accuracy: " + str(vacc))
