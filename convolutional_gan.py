from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU

from simple_gan import *

def build_generator_conv(z_dim):
    model = Sequential()

    model.add(Dense(7*7*256, input_dim = z_dim))
    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, kernel_size = 3, strides = 2, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))

    model.add(Conv2DTranspose(64, kernel_size = 3, strides = 1, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.01))

    model.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = "same"))
    model.add(Activation("tanh"))

    return model

def build_discriminator_conv(img_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides = 2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha = 0.001))

    model.add(Conv2D(64, kernel_size = 3, strides = 2, input_shape = img_shape, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.001))

    model.add(Conv2D(128, kernel_size = 3, strides = 2, input_shape = img_shape, padding = "same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.001))

    model.add(Flatten())
    model.add(Dense(1, activation = "sigmoid"))

    return model


if __name__ == "__main__":

    sample_filename_suffix = "_conv" 
    os.makedirs("./samples{}".format(sample_filename_suffix), exist_ok = True)
    z_dim = 100 
    img_height = 28
    img_width = 28
    channels = 1
    img_shape = (img_height, img_width, channels)

    discriminator_conv = build_discriminator_conv(img_shape)
    discriminator_conv.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    discriminator_conv.trainable = False
    generator_conv = build_generator_conv(z_dim)
    gan_conv = build_gan(generator_conv, discriminator_conv)
    gan_conv.compile(loss = "binary_crossentropy", optimizer=Adam())

    iterations = 20000
    batch_size = 256 
    sample_step = 1000
    metrics = {"losses":[], "accuracies":[], "it_checkpoints":[]}
    train_gan(generator_conv, discriminator_conv, gan_conv, iterations, batch_size, sample_step, z_dim, metrics, sample_filename_suffix)