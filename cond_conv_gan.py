from keras.layers import Input, Embedding, Multiply, Concatenate
from keras.models import Model
from convolutional_gan import *

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf



def build_generator_cond(z_dim):
    z = Input(shape = (z_dim,))
    input_class = Input(shape=(1,), dtype="int32")
    class_embedding = Embedding(class_nb, z_dim, input_length = 1)(input_class)
    class_embedding = Flatten()(class_embedding)
    embedded_class_vector = Multiply()([z, class_embedding])
    generator = build_generator_conv(z_dim)
    img_cond = generator(embedded_class_vector) 
    return Model([z, input_class], img_cond)

def build_discriminator_cond(img_shape):
    img = Input(shape = img_shape)
    input_class = Input(shape=(1,), dtype="int32")
    class_embedding = Embedding(class_nb, np.prod(img_shape), input_length = 1)(input_class)
    class_embedding = Flatten()(class_embedding)
    class_embedding = Reshape(img_shape)(class_embedding)
    embedded_class_tensor = Concatenate(axis = -1)([img, class_embedding])
    discriminator = build_discriminator_conv_cond(img_shape)
    output_class = discriminator(embedded_class_tensor)
    return Model([img, input_class], output_class)

def build_discriminator_conv_cond(img_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides = 2, input_shape=(img_shape[0], img_shape[1], img_shape[2]+1), padding="same"))
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

def build_gan_cond(generator, discriminator):
    z = Input(shape=(z_dim,))
    input_class = Input(shape = (1,))
    img_gen = generator([z, input_class])
    output_class = discriminator([img_gen, input_class])
    model = Model([z, input_class], output_class)
    return model

def train_gan_cond(generator, discriminator, gan, class_nb, iterations, batch_size, sample_step, z_dim, metrics, suffix):
    (X_train, Y_train), (_, _) = fashion_mnist.load_data()
    X_train = X_train/127.5 - 1
    X_train = np.expand_dims(X_train, axis=3)
    real_label = np.ones((batch_size, 1))
    fake_label = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        labels = Y_train[idx]
        z = np.random.normal(0, 1, (batch_size, z_dim))
        fake_imgs = generator.predict([z, labels])

        d_loss_real = discriminator.train_on_batch([real_imgs, labels], real_label)
        d_loss_fake = discriminator.train_on_batch([fake_imgs, labels], fake_label)
        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 1, (batch_size, z_dim))
        labels = np.random.randint(0, class_nb, batch_size).reshape(-1, 1)
        g_loss = gan.train_on_batch([z, labels], real_label)

        if (iteration + 1)%sample_step == 0 or iteration == 0:
            metrics["losses"].append((d_loss[0], g_loss))
            metrics["accuracies"].append(100*d_loss[1])
            metrics["it_checkpoints"].append(iteration + 1)
            print("{} [D loss: {}, acc: {}] [G loss: {}]".format(iteration+1, d_loss[0], 100*d_loss[1], g_loss))
            it1 = iteration + 1
            image_sample_cond(generator, it1, z_dim, suffix, class_nb, batch_size)

def image_sample_cond(generator, it, z_dim, suffix, class_nb, batch_size, img_per_row=4, img_per_col=4):
    z = np.random.normal(0, 1, (img_per_col*img_per_row, z_dim))

    #sample random classes
    #labels = np.random.randint(0, class_nb, batch_size).reshape(-1, 1)

    #sample  a single class k
    #0 T-shirt/top
    #1 Trouser
    #2 Pullover
    #3 Dress
    #4 Coat
    #5 Sandal
    #6 Shirt
    #7 Sneaker
    #8 Bag
    #9 Ankle boot
    k = 1
    labels = np.full((batch_size, 1), k)

    img_gen = generator.predict([z, labels])
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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)

    z_dim = 100 
    img_height = 28
    img_width = 28
    channels = 1
    img_shape = (img_height, img_width, channels)
    class_nb = 10
    sample_filename_suffix = "_cond" 
    os.makedirs("./samples{}".format(sample_filename_suffix), exist_ok = True)

    discriminator_cond = build_discriminator_cond(img_shape)
    discriminator_cond.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    discriminator_cond.trainable = False
    generator_cond = build_generator_cond(z_dim)
    gan_cond = build_gan_cond(generator_cond, discriminator_cond)
    gan_cond.compile(loss = "binary_crossentropy", optimizer=Adam())

    iterations = 20000
    batch_size = 32 
    sample_step = 1000
    metrics = {"losses":[], "accuracies":[], "it_checkpoints":[]}
    train_gan_cond(generator_cond, discriminator_cond, gan_cond, class_nb, iterations, batch_size, sample_step, z_dim, metrics, sample_filename_suffix)