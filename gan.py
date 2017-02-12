import numpy as np
import theano
import theano.tensor as T

class Generator:

    def __init__(self, inputs, hiddens, outputs, N):
        # inputs
        self.x = T.dmatrix('x')

        # weights
        def rand_init(ins, outs, N):
            return np.random.randn(ins, outs) / np.sqrt(N)
        self.w0 = theano.shared(rand_init(inputs, hiddens, N), name="w0")
        self.w1 = theano.shared(rand_init(hiddens, outputs, N), name="w1")

        # equations
        h0 = T.nnet.sigmoid(T.dot(self.x, self.w0))
        h1 = T.nnet.sigmoid(T.dot(h0, self.w1))
        self.p = T.nnet.softmax(h1)

        # compile
        self.generate = theano.function(inputs=[self.x], outputs=[self.p])

feats = 10
hiddens = 20
classes = 30
N = 3
x = np.random.randn(N, feats)
generator = Generator(feats, hiddens, classes, N)
h = generator.generate(x)
print(h)
