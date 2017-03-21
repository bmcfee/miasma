# CREATED: 2/23/17 12:35 by Justin Salamon <justin.salamon@nyu.edu>

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, Convolution1D
from miasma.miasma.layers import SqueezeLayer
# from miasma.miasma.frame_data_generators import get_vad_data_frames
from miasma.miasma.data_generators import get_vad_data
from keras.layers import Input
from keras.models import Model
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
import gzip
import argparse
import keras
import theano
import pescador
import numpy as np


def build_frame_model(tf_rows=288, tf_cols=44, nb_filters=[32, 32],
                      kernel_sizes=[(3, 3), (3, 3)], nb_fullheight_filters=32,
                      loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'], print_model_summary=True,
                      temp_conv=False, freq_conv=False, min_active_frames=10):
    '''
    Build a model that produces frame-level predictions, with no final pool
    layer.

    Parameters
    ----------
    tf_rows
    tf_cols
    nb_filters
    kernel_sizes
    nb_fullheight_filters
    loss
    optimizer
    metrics
    print_model_summary
    temp_conv
    min_active_frames

    Returns
    -------

    '''
    fullheight_kernel_size = (tf_rows, 1)
    if K.image_dim_ordering() == 'th':
        input_shape = (1, tf_rows, tf_cols)
    else:
        input_shape = (tf_rows, tf_cols, 1)
    print('Input shape: {:s}'.format(str(input_shape)))

    assert len(nb_filters) == len(kernel_sizes)

    # MODEL ARCHITECTURE #
    inputs = Input(shape=input_shape, name='input')

    if freq_conv:
        b0 = BatchNormalization(name='b0')(inputs)
        c0 = Convolution2D(1, 3, 1, border_mode='same', activation='relu')(b0)
        b1 = BatchNormalization(name='b1')(c0)
    else:
        b1 = BatchNormalization(name='b1')(inputs)

    c1 = Convolution2D(nb_filters[0], kernel_sizes[0][0], kernel_sizes[0][1],
                       border_mode='same', activation='relu', name='c1')(b1)

    if len(nb_filters) >= 2:
        b2 = BatchNormalization(name='b2')(c1)
        c2 = Convolution2D(nb_filters[1], kernel_sizes[1][0], kernel_sizes[1][1],
                           border_mode='same', activation='relu', name='c2')(b2)
        b3 = BatchNormalization(name='b3')(c2)
    else:
        b3 = BatchNormalization(name='b3')(c1)

    # b3 = BatchNormalization(name='b3')(c2)
    c3 = Convolution2D(nb_fullheight_filters, fullheight_kernel_size[0],
                       fullheight_kernel_size[1], border_mode='valid',
                       activation='relu', name='c3')(b3)

    b4 = BatchNormalization(name='b4')(c3)
    s4 = SqueezeLayer(axis=1, name='s4')(b4)
    c4 = Convolution1D(1, 1, border_mode='valid', activation='sigmoid',
                       name='c4')(s4)

    if temp_conv:
        c5 = Convolution1D(1, min_active_frames, border_mode='same',
                           activation='sigmoid', name='c5')(c4)
        predictions = SqueezeLayer(axis=-1, name='s5')(c5)
    else:
        predictions = SqueezeLayer(axis=-1, name='s5')(c4)

    model = Model(input=inputs, output=predictions)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if print_model_summary:
        model.summary()

    return model


def vad_frame_predictions(expid, pool_layer, split_idx,
                          print_model_summary=False):

    # Load metadata
    root_folder = '/scratch/js7561/datasets/MedleyDB_output'
    model_base_folder = os.path.join(root_folder, 'models')
    model_folder = os.path.join(model_base_folder, expid)
    smp_folder = os.path.join(model_folder, pool_layer)
    metadata_file = os.path.join(smp_folder, '_metadata.json')
    metadata = json.load(open(metadata_file, 'r'))

    # construct kernel_sizes tuple array
    kernel_sizes = []
    for ks in metadata['kernel_sizes']:
        kernel_sizes.append(tuple(ks))

    # Build frame-level model (pooling independent)
    if 'freq_conv' in metadata.keys():
        freq_conv = metadata['freq_conv']
    else:
        freq_conv = False

    model = build_frame_model(
        tf_rows=metadata['tf_rows'],
        tf_cols=metadata['tf_cols'],
        nb_filters=metadata['nb_filters'],
        kernel_sizes=kernel_sizes,
        nb_fullheight_filters=metadata['nb_fullheight_filters'],
        loss=metadata['loss'],
        optimizer=metadata['optimizer'],
        metrics=metadata['metrics'],
        print_model_summary=print_model_summary,
        temp_conv=metadata['temp_conv'],
        freq_conv=freq_conv,
        min_active_frames=metadata['min_active_frames'])

    # Load model weights
    checkpoint_file = os.path.join(
        smp_folder, 'weights_best{:d}.hdf5'.format(split_idx))
    model.load_weights(checkpoint_file, by_name=True)

    # Load validation / test data
    splitfile = '/home/js7561/dev/miasma/data/dataSplits_7_1_2.pkl'
    # (train_generator, X_val, Y_val, ID_val,
    #  X_test, Y_test, ID_test) = (
    #     get_vad_data_frames(
    #         splitfile=splitfile,
    #         split_index=split_idx,
    #         root_folder=root_folder,
    #         augmentations=['original'],
    #         feature='cqt44100_1024_8_36',
    #         activation='vocal_activation44100_1024',
    #         n_bag_frames=metadata['n_bag_frames'],
    #         act_threshold=metadata['act_threshold'],
    #         n_hop_frames=metadata['n_hop_frames'],
    #         batch_size=metadata['batch_size'],
    #         n_samples=metadata['n_samples'],
    #         n_active=metadata['n_active'],
    #         train_id=False,
    #         val_id=True,
    #         test_id=True))

    # Use new generator
    # New code
    (train_generator, X_val, Y_val, ID_val,
     X_test, Y_test, ID_test) = (
        get_vad_data(
            splitfile=splitfile,
            split_index=split_idx,
            root_folder=root_folder,
            augmentations=['original'],
            feature='cqt44100_1024_8_36',
            activation='vocal_activation44100_1024',
            n_bag_frames=metadata['n_bag_frames'],
            min_active_frames=metadata['min_active_frames'],
            act_threshold=metadata['act_threshold'],
            n_hop_frames=metadata['n_hop_frames'],
            batch_size=metadata['batch_size'],
            n_samples=None,
            n_active=metadata['n_active'],
            train_id=False,
            val_id=True,
            test_id=True,
            frame_level=True))

    # Get frame-level predictions
    pred = model.predict(X_test)

    # Compute frame-level metrics
    acc = accuracy_score(
        Y_test.reshape(-1), 1 * (pred.reshape(-1) >= 0.5))
    precision = precision_score(
        Y_test.reshape(-1), 1 * (pred.reshape(-1) >= 0.5))
    recall = recall_score(
        Y_test.reshape(-1), 1 * (pred.reshape(-1) >= 0.5))
    print('Test accuracy: {:.3f}'.format(acc))
    print('Test precision / recall: {:.3f} / {:.3f}'.format(
        precision, recall))

    # Save predictions
    framefolder = os.path.join(smp_folder, 'frame_level')
    if not os.path.isdir(framefolder):
        os.mkdir(framefolder)

    ytestfile = os.path.join(framefolder, 'ytest{:d}.npy.gz'.format(split_idx))
    yprobfile = os.path.join(framefolder, 'yprob{:d}.npy.gz'.format(split_idx))
    yidfile = os.path.join(framefolder, 'yid{:d}.npy.gz'.format(split_idx))

    Y_test.dump(gzip.open(ytestfile, 'wb'))
    pred.dump(gzip.open(yprobfile, 'wb'))
    ID_test.dump(gzip.open(yidfile, 'wb'))


def run_prediction(expid):

    # Print out library versions
    print('keras version: {:s}'.format(keras.__version__))
    print('theano version: {:s}'.format(theano.__version__))
    print('pescador version: {:s}'.format(pescador.__version__))
    print('numpy version: {:s}'.format(np.__version__))

    for pool_layer in ['softmax', 'max', 'mean']:
        print('\n------------- MODEL: {:s}-pooling -------------'.format(
            pool_layer))

        for n, split_idx in enumerate([2, 3, 4, 5, 6]):
            print('\n---------- Split {:d} ----------'.format(split_idx))

            if n == 0:
                print_model_summary = True
            else:
                print_model_summary = False

            vad_frame_predictions(expid, pool_layer, split_idx,
                                  print_model_summary=print_model_summary)

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument('expid', type=str)

        args = parser.parse_args()

        run_prediction(args.expid)

