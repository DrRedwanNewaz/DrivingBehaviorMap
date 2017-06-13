from keras.layers import Dense, Flatten, Input, merge, Dropout, Activation, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l1_l2
import keras.backend as K
import pandas as pd
import numpy as np
from keras_adversarial import AdversarialModel, gan_targets, fix_names, n_choice, simple_bigan
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.layers import BatchNormalization, LeakyReLU
from dataloader import data_base
import h5py
import os



def model_generator(latent_dim, input_shape, hidden_dim=90, reg=lambda: l1_l2(1e-5, 1e-5)):
    return Sequential([
        Dense(int(hidden_dim / 4), name="generator_h1", input_dim=latent_dim, W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(int(hidden_dim / 2), name="generator_h2", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(hidden_dim, name="generator_h3", W_regularizer=reg()),
        LeakyReLU(0.2),
        Dense(np.prod(input_shape), name="generator_x_flat", W_regularizer=reg()),
        Activation('sigmoid'),
        ],
        name="generator")


# Encoder model
def model_encoder(latent_dim, input_shape, hidden_dim=90, reg=lambda: l1(1e-5)):
    x = Input(input_shape, name="x")
    h = Flatten()(x)
    h = Dense(int(hidden_dim), name="encoder_h1", W_regularizer=reg())(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(int(hidden_dim / 4), name="encoder_h2", W_regularizer=reg())(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(int(hidden_dim / 2), name="encoder_h3", W_regularizer=reg())(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(0.2)(h)
    mu = Dense(latent_dim, name="encoder_mu", W_regularizer=reg())(h)
    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", W_regularizer=reg())(h)
    z = merge([mu, log_sigma_sq], mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
              output_shape=lambda x: x[0])

    return Model(x, z, name="encoder")

# Discriminator model
def model_discriminator(latent_dim, input_shape, output_dim=1, hidden_dim=90,
                        reg=lambda: l1_l2(1e-7, 1e-7), dropout=0.5):
    z = Input((latent_dim,))
    x = Input(input_shape, name="x")
    h = merge([z, Flatten()(x)], mode='concat')

    h1 = Dense(hidden_dim, name="discriminator_h1", W_regularizer=reg())
    b1 = BatchNormalization()
    h2 = Dense(hidden_dim, name="discriminator_h2", W_regularizer=reg())
    b2 = BatchNormalization()
    h3 = Dense(hidden_dim, name="discriminator_h3", W_regularizer=reg())
    b3 = BatchNormalization()
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())

    # training model uses dropout
    _h = h
    _h = Dropout(dropout)(LeakyReLU(0.2)((b1(h1(_h)))))
    _h = Dropout(dropout)(LeakyReLU(0.2)((b2(h2(_h)))))
    _h = Dropout(dropout)(LeakyReLU(0.2)((b3(h3(_h)))))
    ytrain = y(_h)
    mtrain = Model([z, x], ytrain, name="discriminator_train")

    # testing model does not use dropout
    _h = h
    _h = LeakyReLU(0.2)((b1(h1(_h))))
    _h = LeakyReLU(0.2)((b2(h2(_h))))
    _h = LeakyReLU(0.2)((b3(h3(_h))))
    ytest = y(_h)
    mtest = Model([z, x], ytest, name="discriminator_test")

    return mtrain, mtest

def driver_gan(path, adversarial_optimizer):
    # z \in R^100
    latent_dim = 3
    # x \in R^{28x28}
    input_shape = (15, 6)

    # generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape)
    # autoencoder (x -> x')
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (x -> y)
    discriminator_train, discriminator_test = model_discriminator(latent_dim, input_shape)
    # bigan (z, x - > yfake, yreal)
    bigan_generator = simple_bigan(generator, encoder, discriminator_test)
    bigan_discriminator = simple_bigan(generator, encoder, discriminator_train)
    # z generated on GPU based on batch dimension of x
    x = bigan_generator.inputs[1]
    z = normal_latent_sampling((latent_dim,))(x)
    # eliminate z from inputs
    bigan_generator = Model([x], fix_names(bigan_generator([z, x]), bigan_generator.output_names))
    bigan_discriminator = Model([x], fix_names(bigan_discriminator([z, x]), bigan_discriminator.output_names))

    # Merging encoder weights and generator weights
    generative_params = generator.trainable_weights + encoder.trainable_weights

    # print summary of models
    generator.summary()
    encoder.summary()
    discriminator_train.summary()
    bigan_discriminator.summary()
    autoencoder.summary()

    # build adversarial model
    model = AdversarialModel(player_models=[bigan_generator, bigan_discriminator],
                             player_params=[generative_params, discriminator_train.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[Adam(1e-7, decay=1e-7), Adam(1e-6, decay=1e-7)],
                              loss='binary_crossentropy')

    # load driver data
    train_dataset = [1,2,5]
    test_dataset = [3,4]
    train_reader = data_base(train_dataset)
    test_reader = data_base(test_dataset)
    xtrain, xtest = train_reader.read_files(),test_reader.read_files()
    # ---------------------------------------------------------------------------------
    # callback for image grid of generated samples
    def generator_sampler():
        zsamples = np.random.normal(size=(1 * 1, latent_dim))  #---------------------------------> (10,10)
        return generator.predict(zsamples).reshape((1, 1, 15, 6))# confused ***********************************default (10,10,28,28)


    # callback for image grid of autoencoded samples
    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10) # the number of testdata set
        xrep = np.repeat(xsamples, 5, axis=0) # the number of train dataset
        xgen = autoencoder.predict(xrep).reshape((1, 1, 15, 6))
        xsamples = xsamples.reshape((1, 1, 15, 6))
        x = np.concatenate((xsamples, xgen), axis=1)
        return x


    # train network
    y = gan_targets(xtrain.shape[0])
    ytest = gan_targets(xtest.shape[0])
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest),
                        nb_epoch=25, batch_size=10, verbose=0)

    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, "history.csv"))

    # save model
    encoder.save(os.path.join(path, "encoder.h5"))
    generator.save(os.path.join(path, "generator.h5"))
    discriminator_train.save(os.path.join(path, "discriminator.h5"))


def main():
    driver_gan("output/result", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()