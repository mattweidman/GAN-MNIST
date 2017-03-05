# using tutorial from http://www.kdnuggets.com/2016/07/mnist-generative-adversarial-model-keras.html

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import keras.layers as kl
import keras.layers.advanced_activations as kla
import keras.layers.convolutional as klconv
import keras.layers.core as klc
import keras.layers.normalization as kln

import keras.models as km

import keras.optimizers as ko

import mnist_parse

# parameters
n_train = 50000
classes = 10
batch_size = 100
noise_size = 100
dropout_rate = 0.25

# optimizers
g_opt = ko.Adam(lr=1e-4)
d_opt = ko.Adam(lr=1e-3)

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
def gen_noise(batch_size, d):
    return np.random.uniform(0, 1, size=[batch_size, d])

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
generator.compile(loss='binary_crossentropy', optimizer=g_opt)

# discriminator
d_input = kl.Input(shape=[1,img_rows,img_cols])
H = klconv.Convolution2D(256, 5, 5, subsample=(2,2), border_mode='same',
        activation='relu')(d_input)
H = kla.LeakyReLU(0.2)(H)
H = klc.Dropout(dropout_rate)(H)
H = klconv.Convolution2D(512, 5, 5, subsample=(2,2), border_mode='same',
        activation='relu')(H)
H = kla.LeakyReLU(0.2)(H)
H = klc.Dropout(dropout_rate)(H)
H = klc.Flatten()(H)
H = klc.Dense(256)(H)
H = kla.LeakyReLU(0.2)(H)
H = klc.Dropout(dropout_rate)(H)
d_V = klc.Dense(2, activation='softmax')(H)
discriminator = km.Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=d_opt)

# stack generator and discriminator
gan_input = kl.Input(shape=[noise_size])
H = generator(gan_input)
gan_v = discriminator(H)
GAN = km.Model(gan_input, gan_v)
GAN.compile(loss='categorical_crossentropy', optimizer=g_opt)
GAN.summary()

# make data for pretraining
n_pretrain = 1000
noise = gen_noise(n_pretrain, noise_size)
gen_imgs = generator.predict(noise)
indices = np.random.randint(n_train, size=(n_pretrain))
real_imgs = trainX[indices]
x_pretrain = np.concatenate((real_imgs, gen_imgs))
y_pretrain = np.zeros((2*n_pretrain, 2))
y_pretrain[:n_pretrain, 1] = 1
y_pretrain[n_pretrain:, 0] = 1

# pretrain discriminator and find accuracy
print("pretraining...")
discriminator.fit(x_pretrain, y_pretrain, nb_epoch=1, batch_size=batch_size)
y_hat = discriminator.predict(x_pretrain)
y_hat_indices = np.argmax(y_hat, axis=1)
y_indices = np.argmax(y_pretrain, axis=1)
acc = (y_hat_indices == y_indices).mean() * 100.0
print("Discriminator pretraining accuracy: " + str(acc))

# display losses on graph
def plot_loss(losses_d, losses_g):
    plt.figure(figsize=(10,8))
    plt.plot(losses_d, label='discriminative loss')
    plt.plot(losses_g, label='generative loss')
    plt.legend()
    plt.show()

# display some generated images
def plot_images(n_images=16, dim=(4,4), figsize=(10,10)):
    noise = noise_gen(n_images, noise_size)
    imgs = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(n_images):
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(imgs[i,0,:,:])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# sets whether a net can currently be trained
def make_trainable(net, val):
    net.trainable = val
    for layer in net.layers:
        layer.trainable = val

# list of losses to be used by the plot
losses_d = []
losses_g = []

# training function
def train(n_epochs=2000, plot_freq=25):
    for i in range(n_epochs):

        # run generator
        noise = gen_noise(batch_size, noise_size)
        gen_imgs = generator.predict(noise)

        # prepare data for discriminator
        indices = np.random.randint(n_train, size=(batch_size))
        real_imgs = trainX[indices]
        X = np.concatenate((real_imgs, gen_imgs))
        Y = np.zeros((2*batch_size, 2))
        Y[:batch_size, 1] = 1
        Y[batch_size:, 1] = 1

        # train discriminator
        make_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch(X,Y)
        losses_d.append(d_loss)

        # prepare data for generator
        noise = gen_noise(batch_size, noise_size)
        y2 = np.zeros((batch_size, 2))
        y2[:,1] = 1

        # train generator
        make_trainable(discriminator, False)
        g_loss = GAN.train_on_batch(noise, y2)
        losses_g.append(g_loss)

        # update plots
        if i % plot_freq == plot_freq-1:
            plot_loss(losses_d, losses_g)
            plot_images()

        print("epoch: " + str(i) + ", d_loss: " + str(d_loss) + ", g_loss: " \
                + str(g_loss))

print("training...")

# train with default learning rate
train(n_epochs=6000)

# lower learning rate and train again
g_opt.lr.set_value(1e-5)
d_opt.lr.set_value(1e-4)
train()

# and one more time
g_opt.lr.set_value(1e-6)
d_opt.lr.set_value(1e-5)
train()

# plot losses one final time
plot_loss(losses_d, losses_g)

def save_images(predict, img_name):
    imgs = predict.reshape(n_train, img_rows, img_cols)
    for i in range(len(imgs)):
        scipy.misc.imsave(img_name + str(i) + ".png", imgs[i])

