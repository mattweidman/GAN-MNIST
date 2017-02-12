import numpy as np
import theano
import theano.tensor as T

import mnist_parse

# parameters
lr = 0.1

class Generator:

    def __init__(self, inputs, hiddens, outputs, N):
        # inputs
        self.x = T.dmatrix('xG')

        # weights
        def rand_init(ins, outs, N):
            return np.random.randn(ins, outs) / np.sqrt(N)
        self.w0 = theano.shared(rand_init(inputs, hiddens, N), name="w0G")
        self.w1 = theano.shared(rand_init(hiddens, outputs, N), name="w1G")

        # equations
        h0 = T.nnet.sigmoid(T.dot(self.x, self.w0))
        h1 = T.nnet.sigmoid(T.dot(h0, self.w1))
        self.p = T.nnet.softmax(h1)

        # compile
        self.generate = theano.function(inputs=[self.x], outputs=[self.p])

class Discriminator:

    def __init__(self, inputs, hiddens, N):
        self.x = T.dmatrix('xD')

        # weights
        def rand_init(ins, outs, N):
            return np.random.randn(ins, outs) / np.sqrt(N)
        self.w0 = theano.shared(rand_init(inputs, hiddens, N), name="w0D")
        self.w1 = theano.shared(rand_init(hiddens, 1, N), name="w1D")

        # equations
        h0 = T.nnet.sigmoid(T.dot(self.x, self.w0))
        self.p = T.nnet.sigmoid(T.dot(h0, self.w1))

        # compile
        classify = theano.function(inputs=[self.x], outputs=[self.p])

class GAN:

    def __init__(self, inputs, hiddenG, outputs, hiddenD, N_noise, N_data):
        # init generator and discriminator
        self.G = Generator(inputs, hiddenG, outputs, N_noise)
        self.D = Discriminator(outputs, hiddenD, N_noise + N_data)

        # variables
        self.noise = T.dmatrix('noise')
        self.data = T.dmatrix('data')
        self.concat_indx = T.scalar('concat_indx')

        # equations
        gen_data = self.G.p
        all_data = T.concatenate((gen_data, self.data), axis=0)
        classes = self.D.p # TODO: make this dependent on all_data
        self.loss_D = -0.5 * T.log(classes[:self.concat_indx,:]).mean() \
                -0.5 * T.log(1-classes[self.concat_indx:,:]).mean()
        self.loss_G = -0.5 * T.log(classes[self.concat_indx:,:]).mean()
        self.gw0D = T.grad(self.loss_D, self.D.w0)
        self.gw1D = T.grad(self.loss_D, self.D.w1)
        self.gw0G = T.grad(self.loss_G, self.G.w0)
        self.gw1G = T.grad(self.loss_G, self.G.w1)

        # compile
        train_D = theano.function(inputs=[self.x,self.concat_indx],
                outputs=[self.loss_D],
                updates=[(self.D.w0, self.D.w0 - lr*self.gw0D),
                    (self.D.w1, self.D.w1 - lr*self.gw1D)])
        train_G = theano.function(inputs=[self.x,self.concat_indx],
                outputs=[self.loss_G],
                updates=[(self.G.w0, self.G.w0 - lr*self.gw0G),
                    (self.G.w1, self.G.w1 - lr*self.gw1G)])

N = 1000
noise_feats = 100
hiddens_G = 300
hiddens_D = 300
rawX = mnist_parse.getImages()[:N]

# wrangle data
def x_wrangle(x_in):
    x = x_in.reshape(x_in.shape[0], x_in.shape[1]*x_in.shape[2])
    x = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)
    #x -= x.mean(axis=1)[:, np.newaxis]
    #x /= x.std(axis=1)[:, np.newaxis]
    return x
trainX = x_wrangle(rawX)
feats = trainX.shape[1]

# create GAN
gan = GAN(noise_feats, hiddens_G, feats, hiddens_D, N, N)

#feats = 10
#hiddens = 20
#classes = 30
#N = 3
#x = np.random.randn(N, feats)
#generator = Generator(feats, hiddens, classes, N)
#h = generator.generate(x)
#print(h)
#print(h[0].shape)
