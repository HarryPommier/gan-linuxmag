import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
import os


def build_generator(img_shape, z_dim):
    model = Sequential()
    model.add(Dense(128, input_dim = z_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(28*28*1, input_dim = z_dim, activation = "tanh"))
    model.add(Reshape(img_shape))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation = "sigmoid"))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

def train_gan(generator, discriminator, gan, iterations, batch_size, sample_step, z_dim, metrics, suffix):
    (X_train, _), (_, _) = fashion_mnist.load_data()
    X_train = X_train/127.5 - 1
    X_train = np.expand_dims(X_train, axis=3)
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        z = np.random.normal(0, 1, (batch_size, z_dim))
        fake_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(real_imgs, real_label)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss, accuracy = 0.5*np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, z_dim))
        g_loss = gan.train_on_batch(z, real_label)

        if (iteration + 1)%sample_step == 0 or iteration == 0:
            metrics["losses"].append((d_loss, g_loss))
            metrics["accuracies"].append(100*accuracy)
            metrics["it_checkpoints"].append(iteration + 1)
            print("{} [D loss: {}, acc: {}] [G loss: {}]".format(iteration+1, d_loss, 100*accuracy, g_loss))
            it1 = iteration + 1
            image_sample(generator, it1, z_dim, suffix)

def image_sample(generator, it, z_dim, suffix, img_per_row=4, img_per_col=4):
    z = np.random.normal(0, 1, (img_per_col*img_per_row, z_dim))
    img_gen = generator.predict(z)
    img_gen = 0.5*img_gen + 0.5
    fig, ax = plt.subplots(img_per_row, img_per_col, figsize=(img_per_row, img_per_col), sharex = True, sharey = True)
    cpt = 0
    for i in range(img_per_row):
        for j in range(img_per_col):
            ax[i, j].imshow(img_gen[cpt,:,:,0], cmap = "gray")
            ax[i, j].axis("off")
            cpt += 1
    plt.savefig("samples{}/it_{}.jpg".format(suffix, it), format="jpg")


if __name__ == "__main__":
    sample_filename_suffix = "_simple" 
    os.makedirs("./samples{}".format(sample_filename_suffix), exist_ok = True)
    z_dim = 100 
    img_height = 28
    img_width = 28
    channels = 1
    img_shape = (img_height, img_width, channels)
    
    discriminator = build_discriminator(img_shape)
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    discriminator.trainable = False
    generator = build_generator(img_shape, z_dim)
    gan = build_gan(generator, discriminator)
    gan.compile(loss = "binary_crossentropy", optimizer=Adam())

    iterations = 20000
    batch_size = 256 
    sample_step = 1000
    metrics = {"losses":[], "accuracies":[], "it_checkpoints":[]}
    train_gan(generator, discriminator, gan, iterations, batch_size, sample_step, z_dim, metrics, sample_filename_suffix)