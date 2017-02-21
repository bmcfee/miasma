# CREATED: 2/15/17 5:28 PM by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
import os
import json
from miasma.miasma.layers import SoftMaxPool, SqueezeLayer
from miasma.miasma.data_generators import get_vad_data
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import pickle
import theano
import keras
import pescador
from sklearn.metrics import accuracy_score
import gzip

np.random.seed(1337)  # for reproducibility


def build_smp_model(tf_rows=288, tf_cols=44, nb_filters=[32, 32],
                    kernel_sizes=[(3, 3), (3, 3)], nb_fullheight_filters=32,
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

    assert len(nb_filters) == len(kernel_sizes)

    b1 = BatchNormalization(name='b1')(inputs)
    c1 = Convolution2D(nb_filters[0], kernel_sizes[0][0], kernel_sizes[0][1],
                       border_mode='same', activation='relu', name='c1')(b1)

    # b2 = BatchNormalization(name='b2')(c1)
    # c2 = Convolution2D(nb_filters[1], kernel_sizes[1][0], kernel_sizes[1][1],
    #                    border_mode='same', activation='relu', name='c2')(b2)

    b3 = BatchNormalization(name='b3')(c1)
    c3 = Convolution2D(nb_fullheight_filters, fullheight_kernel_size[0],
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
                                  verbose=verbose,
                                  validation_data=(X_val, Y_val),
                                  callbacks=[checkpointer])

    return history


def run_experiment(expid, n_bag_frames=44, min_active_frames=10,
                   act_threshold=0.5, n_hop_frames=22, batch_size=32,
                   n_samples=None, n_active=1000, samples_per_epoch=1024,
                   nb_epochs=50, verbose=1,
                   tf_rows=288, tf_cols=44, nb_filters=[32, 32],
                   kernel_sizes=[(3, 3), (3, 3)], nb_fullheight_filters=32,
                   loss='binary_crossentropy', optimizer='adam',
                   metrics=['accuracy']):

    root_folder = '/scratch/js7561/datasets/MedleyDB_output'
    model_base_folder = os.path.join(root_folder, 'models')
    splitfile = '/home/js7561/dev/miasma/data/dataSplits_7_1_2.pkl'
    split_indices = [2, 3, 4, 5, 6]

    # Create a folder for this experiment
    model_folder = os.path.join(model_base_folder, expid)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    # --- SMP MODEL ---
    smp_folder = os.path.join(model_folder, 'smp')
    if not os.path.isdir(smp_folder):
        os.mkdir(smp_folder)

    # Save experiment metadata
    metadata_file = os.path.join(smp_folder, '_metadata.json')
    metadata = {
        'root_folder': root_folder,
        'model_base_folder': model_base_folder,
        'splitfile': splitfile,
        'split_indices': split_indices,
        'model_folder': model_folder,
        'smp_folder': smp_folder,
        'expid': expid,
        'n_bag_frames': n_bag_frames,
        'min_active_frames': min_active_frames,
        'act_threshold': act_threshold,
        'n_hop_frames': n_hop_frames,
        'batch_size': batch_size,
        'n_samples': n_samples,
        'n_active': n_active,
        'samples_per_epoch': samples_per_epoch,
        'nb_epochs': nb_epochs,
        'verbose': verbose,
        'tf_rows': tf_rows,
        'tf_cols': tf_cols,
        'nb_filters': nb_filters,
        'kernel_sizes': kernel_sizes,
        'nb_fullheight_filters': nb_fullheight_filters,
        'loss': loss,
        'optimizer': optimizer,
        'metrics': metrics,
        'theano_version': theano.__version__,
        'keras_version:': keras.__version__,
        'numpy_version': np.__version__,
        'pescador_version': pescador.__version__}

    print(metadata)
    json.dump(metadata, open(metadata_file, 'w'), indent=2)

    # Repeat for 5 train/validate/test splits
    for split_idx in split_indices:

        # Build model
        model = build_smp_model(
            tf_rows=tf_rows, tf_cols=tf_cols, nb_filters=nb_filters,
            kernel_sizes=kernel_sizes,
            nb_fullheight_filters=nb_fullheight_filters, loss=loss,
            optimizer=optimizer, metrics=metrics)

        # Load data
        train_generator, X_val, Y_val, X_test, Y_test = (
            get_vad_data(
                splitfile=splitfile,
                split_index=split_idx,
                root_folder=root_folder,
                augmentations=['original'],
                feature='cqt44100_1024_8_36',
                activation='vocal_activation44100_1024',
                n_bag_frames=n_bag_frames,
                min_active_frames=min_active_frames,
                act_threshold=act_threshold,
                n_hop_frames=n_hop_frames,
                batch_size=batch_size,
                n_samples=n_samples,
                n_active=n_active))

        checkpoint_file = os.path.join(
            smp_folder, 'weights{:d}.hdf5'.format(split_idx))

        # Train
        history = fit_model(model, checkpoint_file, train_generator, X_val,
                            Y_val, samples_per_epoch=samples_per_epoch,
                            nb_epochs=nb_epochs, verbose=verbose)

        # Test
        pred = model.predict(X_test)
        pred = pred.reshape((-1))
        acc = accuracy_score(Y_test, 1 * (pred >= 0.5))
        print('Test accuracy: {:.3f}'.format(acc))

        # Save Y_test and predictions
        ytestfile = os.path.join(
            smp_folder, 'ytest{:d}.npy.gz'.format(split_idx))
        yprobfile = os.path.join(
            smp_folder, 'yprob{:d}.npy.gz'.format(split_idx))

        Y_test.dump(gzip.open(ytestfile, 'wb'))
        pred.dump(gzip.open(yprobfile, 'wb'))

        # Save history
        history_file = os.path.join(
            smp_folder, 'history{:d}.pkl'.format(split_idx))
        pickle.dump(history, open(history_file, 'wb'))

