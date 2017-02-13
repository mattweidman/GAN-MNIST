import numpy as np
import scipy.misc
import theano
import theano.tensor as T

import mnist_parse

class GAN:

    def __init__(self, feats_G, hiddens_G, feats_D, hiddens_D, N_G, N_D, lr):
        """
        feats_G: number of features as input to generator
        hiddens_G: number of hidden units in generator
        feats_D: number of features in training data
        hiddens_D: number of hidden units in discriminator
        N_G: number of examples as input to generator
        N_D: number of examples as input to discriminator
        lr: learning rate
        """

        # variables
        self.noise = T.dmatrix('noise')
        self.data = T.dmatrix('data')
        self.div_indx = T.scalar('div_indx', dtype='int32')

        def rand_n(ins, outs, N):
            return np.random.randn(ins, outs) / np.sqrt(N)
        self.w0G = theano.shared(rand_n(feats_G, hiddens_G, N_G), name="w0G")
        self.w1G = theano.shared(rand_n(hiddens_G, feats_D, N_G), name="w1G")
        self.w0D = theano.shared(rand_n(feats_D, hiddens_D, N_D), name="w0D")
        self.w1D = theano.shared(rand_n(hiddens_D, 1, N_D), name="w1D")

        # equations
        h0G = T.nnet.sigmoid(T.dot(self.noise, self.w0G))
        h1G = T.nnet.sigmoid(T.dot(h0G, self.w1G))
        self.gen = h1G

        all_data = T.concatenate((self.gen, self.data), axis=0)
        h0D = T.nnet.sigmoid(T.dot(all_data, self.w0D))
        self.p = T.nnet.sigmoid(T.dot(h0D, self.w1D))

        self.loss_D = -0.5 * T.log(self.p[:self.div_indx,:]).mean() \
                -0.5 * T.log(self.p[self.div_indx:,:]).mean()
        self.loss_G = -0.5 * T.log(self.p[:self.div_indx,:]).mean()

        self.gw0D = T.grad(self.loss_D, self.w0D)
        self.gw1D = T.grad(self.loss_D, self.w1D)
        self.gw0G = T.grad(self.loss_G, self.w0G)
        self.gw1G = T.grad(self.loss_G, self.w1G)

        # compile
        self.train_D = theano.function(inputs=[self.noise,
            self.data, self.div_indx],
                outputs=[self.loss_D],
                updates=[(self.w0D, self.w0D - lr*self.gw0D),
                    (self.w1D, self.w1D - lr*self.gw1D)])
        self.train_G = theano.function(inputs=[self.noise,
            self.data, self.div_indx],
                outputs=[self.loss_G],
                updates=[(self.w0G, self.w0G - lr*self.gw0G),
                    (self.w1G, self.w1G - lr*self.gw1G)])
        self.generate = theano.function(inputs=[self.noise],
                outputs=[self.gen])

# parameters
feats_G = 100
hiddens_G = 300
hiddens_D = 300
N = 1000
lr = 0.1
epochs = 100

# load data
rawX = mnist_parse.getImages()[:N]
def x_wrangle(x_in):
    x = x_in / 255
    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    x = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)
    return x
trainX = x_wrangle(rawX)
feats_D = trainX.shape[1]

# initialize
gan = GAN(feats_G, hiddens_G, feats_D, hiddens_D, N, N, lr)

# train
for i in range(epochs):
    noise = np.random.randn(N, feats_G)
    loss_D = gan.train_D(noise, trainX, N)
    loss_G = gan.train_G(noise, trainX, N)
    if i % 10 == 0:
        print(str(loss_D[0]) + ", " + str(loss_G[0]))

# try generating
noise = np.random.randn(1, feats_G)
generated = gan.generate(noise)[0]
img = generated[:,:-1].reshape(28,28)
scipy.misc.imsave("mnist_gen.png", img)
