# CREATED: 2/15/17 5:28 PM by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np

from miasma.layers import SoftMaxPool, SqueezeLayer
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

np.random.seed(1337)  # for reproducibility


def build_model(tf_rows=288, tf_cols=44, nb_filters=32,
                nb_filters_fullheight=16, kernel_size=(3, 3),
                loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy']):

    fullheight_kernel_size = (tf_rows, 1)
    if K.image_dim_ordering() == 'th':
        input_shape = (1, tf_rows, tf_cols)
    else:
        input_shape = (tf_rows, tf_cols, 1)
    print('Input shape: {:s}'.format(str(input_shape)))

    # MODEL ARCHITECTURE #
    inputs = Input(shape=input_shape, name='input')

    b1 = BatchNormalization(name='b1')(inputs)
    c1 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                       border_mode='same', activation='relu', name='c1')(b1)

    b2 = BatchNormalization(name='b2')(c1)
    c2 = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                       border_mode='same', activation='relu', name='c2')(b2)

    b3 = BatchNormalization(name='b3')(c2)
    c3 = Convolution2D(nb_filters_fullheight, fullheight_kernel_size[0],
                       fullheight_kernel_size[1], border_mode='valid',
                       activation='relu', name='c3')(b3)

    b4 = BatchNormalization(name='b4')(c3)
    s4 = SqueezeLayer(axis=1, name='s4')(b4)
    c4 = Convolution1D(1, 1, border_mode='valid', activation='sigmoid',
                       name='c4')(s4)

    s5 = SqueezeLayer(axis=-1, name='s5')(c4)
    predictions = SoftMaxPool(name='smp')(s5)

    model = Model(input=inputs, output=predictions)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.summary()

    return model


def fit_model(model, checkpoint_file, train_generator, X_val, Y_val,
              samples_per_epoch=1024, nb_epochs=50, verbose=1):

    checkpointer = ModelCheckpoint(filepath=checkpoint_file, verbose=0,
                                   save_best_only=True)

    history = model.fit_generator(train_generator,
                                  samples_per_epoch,
                                  nb_epochs,
                                  verbose=1,
                                  validation_data=(X_val, Y_val),
                                  callbacks=[checkpointer])

    return history

