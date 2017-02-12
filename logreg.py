import numpy as np
import theano
import theano.tensor as T

import mnist_parse

# parameters
N = 1000
lr = 3
epochs = 0

# import data
rawX = mnist_parse.getImages()[:N]
rawY = mnist_parse.getLabels()[:N]

trainX = rawX.reshape(N, rawX.shape[1]*rawX.shape[2])
trainX = np.concatenate((trainX, np.ones((N,1))), axis=1)
feats = trainX.shape[1]

classes = 10
trainY = np.zeros((N, classes))
trainY[np.arange(N), rawY] = 1

# initialize variables
x = T.dmatrix("x")
y = T.dmatrix("y")
w = theano.shared(np.random.randn(feats, classes), name="w")

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
print(acc)
