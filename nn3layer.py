import numpy as np
import theano
import theano.tensor as T

import mnist_parse

# parameters
N = 50000
valN = 10000
lr = 10
lr_decay = 1 #0.999
epochs = 10000
hiddens = 25
mb_size = 100

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
w0 = theano.shared((np.random.randn(feats, hiddens)/np.sqrt(N)), name="w0")
w1 = theano.shared((np.random.randn(hiddens, classes)/np.sqrt(N)), name="w1")

# equations
h0 = T.nnet.sigmoid(T.dot(x,w0))
h1 = T.nnet.sigmoid(T.dot(h0,w1))
p = T.nnet.softmax(h1)
loss = (-y*T.log(p)).mean()
gw0 = T.grad(loss, w0)
gw1 = T.grad(loss, w1)
accuracy = T.eq(T.argmax(p, axis=1), T.argmax(y, axis=1)).mean()

# compile
train = theano.function(inputs=[x,y], outputs=[loss],
        updates=[(w0, w0-lr*gw0), (w1, w1-lr*gw1)])
predict = theano.function(inputs=[x], outputs=[p])
evaluate = theano.function(inputs=[x,y], outputs=[accuracy])

# train
for i in range(epochs):
    indices = np.random.randint(0, high=N, size=mb_size)
    loss = train(trainX[indices,:], trainY[indices])
    lr *= lr_decay
    if (i % 1000 == 0):
        print(loss[0], lr)
acc = evaluate(trainX, trainY)[0]
print("training accuracy: " + str(acc))
vacc = evaluate(valX, valY)[0]
print("validation accuracy: " + str(vacc))
