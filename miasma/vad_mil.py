# CREATED: 2/15/17 5:28 PM by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
import os
import json
from miasma.miasma.layers import SoftMaxPool, SqueezeLayer, BagToBatchLayer
from miasma.miasma.data_generators import get_vad_data
# from miasma.miasma.frame_data_generators import get_vad_data_frames
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, Convolution1D
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint
import pickle
import theano
import keras
import pescador
from sklearn.metrics import accuracy_score
import gzip
import sys
import argparse
import time


np.random.seed(1337)  # for reproducibility
sys.setrecursionlimit(50000)  # to pickle keras history objects


def build_model(tf_rows=288, tf_cols=44, nb_filters=[32, 32],
                kernel_sizes=[(3, 3), (3, 3)], nb_fullheight_filters=32,
                loss='binary_crossentropy', optimizer='adam',
                metrics=['accuracy'], pool_layer='softmax',
                print_model_summary=True, temp_conv=False, freq_conv=False,
                min_active_frames=10, dropout=False, reg_W=False,
                reg_activity=False):

    fullheight_kernel_size = (tf_rows, 1)
    if K.image_dim_ordering() == 'th':
        input_shape = (1, tf_rows, tf_cols)
    else:
        input_shape = (tf_rows, tf_cols, 1)
    print('Input shape: {:s}'.format(str(input_shape)))

    assert len(nb_filters) == len(kernel_sizes)

    if reg_W:
        reg_W_func = l2(0.01)
    else:
        reg_W_func = None

    if reg_activity:
        reg_activity_func = l1(0.01)
    else:
        reg_activity_func = None

    # MODEL ARCHITECTURE
    inputs = Input(shape=input_shape, name='input')

    if freq_conv:
        b0 = BatchNormalization(name='b0')(inputs)
        c0 = Convolution2D(1, 3, 1, border_mode='same', activation='relu',
                           name='c0', W_regularizer=reg_W_func,
                           activity_regularizer=reg_activity_func)(b0)
        b1 = BatchNormalization(name='b1')(c0)
    else:
        b1 = BatchNormalization(name='b1')(inputs)

    c1 = Convolution2D(nb_filters[0], kernel_sizes[0][0], kernel_sizes[0][1],
                       border_mode='same', activation='relu', name='c1',
                       W_regularizer=reg_W_func,
                       activity_regularizer=reg_activity_func)(b1)

    if len(nb_filters) >= 2:
        b2 = BatchNormalization(name='b2')(c1)
        c2 = Convolution2D(nb_filters[1], kernel_sizes[1][0],
                           kernel_sizes[1][1],
                           border_mode='same', activation='relu', name='c2',
                           W_regularizer=reg_W_func,
                           activity_regularizer=reg_activity_func)(b2)
        b3 = BatchNormalization(name='b3')(c2)
    else:
        b3 = BatchNormalization(name='b3')(c1)

    # b3 = BatchNormalization(name='b3')(c2)
    c3 = Convolution2D(nb_fullheight_filters, fullheight_kernel_size[0],
                       fullheight_kernel_size[1], border_mode='valid',
                       activation='relu', name='c3',
                       W_regularizer=reg_W_func,
                       activity_regularizer=reg_activity_func)(b3)

    b4 = BatchNormalization(name='b4')(c3)
    s4 = SqueezeLayer(axis=1, name='s4')(b4)
    c4 = Convolution1D(1, 1, border_mode='valid', activation='sigmoid',
                       name='c4', W_regularizer=reg_W_func,
                       activity_regularizer=reg_activity_func)(s4)

    if dropout:
        c4 = Dropout(0.5, name='c4dropout')(c4)

    if pool_layer == 'softmax':
        if temp_conv:
            c5 = Convolution1D(1, min_active_frames, border_mode='same',
                               activation='sigmoid', name='c5',
                               W_regularizer=reg_W_func,
                               activity_regularizer=reg_activity_func)(c4)
            s5 = SqueezeLayer(axis=-1, name='s5')(c5)
        else:
            s5 = SqueezeLayer(axis=-1, name='s5')(c4)
        predictions = SoftMaxPool(name='softmax-pool')(s5)

    elif pool_layer == 'max':
        if temp_conv:
            c5 = Convolution1D(1, min_active_frames, border_mode='same',
                               activation='sigmoid', name='c5',
                               W_regularizer=reg_W_func,
                               activity_regularizer=reg_activity_func)(c4)
            p5 = MaxPooling1D(pool_length=tf_cols, stride=None,
                              border_mode='valid', name='max-pool')(c5)
        else:
            p5 = MaxPooling1D(pool_length=tf_cols, stride=None,
                              border_mode='valid', name='max-pool')(c4)
        predictions = SqueezeLayer(axis=-1, name='s5')(p5)

    elif pool_layer == 'mean':
        if temp_conv:
            c5 = Convolution1D(1, min_active_frames, border_mode='same',
                               activation='sigmoid', name='c5',
                               W_regularizer=reg_W_func,
                               activity_regularizer=reg_activity_func)(c4)
            p5 = AveragePooling1D(pool_length=tf_cols, stride=None,
                                  border_mode='valid', name='mean-pool')(c5)
        else:
            p5 = AveragePooling1D(pool_length=tf_cols, stride=None,
                                  border_mode='valid', name='mean-pool')(c4)
        predictions = SqueezeLayer(axis=-1, name='s5')(p5)

    elif pool_layer == 'none':
        # predictions = BagToBatchLayer(name='none-pool')(c4)
        predictions = SqueezeLayer(axis=-1, name='s5')(c4)

    else:
        print('Unrecognized pooling, using softmax')
        s5 = SqueezeLayer(axis=-1, name='s5')(c4)
        predictions = SoftMaxPool(name='pool')(s5)

    model = Model(input=inputs, output=predictions)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    if print_model_summary:
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


def fit_model_valgenerator(model, checkpoint_file, train_generator,
                           validate_generator, samples_per_epoch=1024,
                           nb_epochs=50, verbose=1, nb_val_samples=4000):

    checkpointer = ModelCheckpoint(filepath=checkpoint_file, verbose=0,
                                   save_best_only=True)

    history = model.fit_generator(train_generator,
                                  samples_per_epoch,
                                  nb_epochs,
                                  verbose=verbose,
                                  validation_data=validate_generator,
                                  callbacks=[checkpointer],
                                  nb_val_samples=nb_val_samples)

    return history


def run_experiment(expid, n_bag_frames=44, min_active_frames=10,
                   act_threshold=0.5, n_hop_frames=22, batch_size=32,
                   n_samples=None, n_active=128, samples_per_epoch=1024,
                   nb_epochs=50, verbose=1,
                   tf_rows=288, tf_cols=44, nb_filters=[32, 32],
                   kernel_sizes=[(3, 3), (3, 3)], nb_fullheight_filters=32,
                   loss='binary_crossentropy', optimizer='adam',
                   metrics=['accuracy', 'precision', 'recall'],
                   split_indices=[0, 1, 2, 3, 4],
                   pool_layers=['max', 'mean', 'softmax'],
                   temp_conv=False, freq_conv=False, augs=['original'],
                   dropout=False, reg_W=False, reg_activity=False):

    # Print out library versions
    print('keras version: {:s}'.format(keras.__version__))
    print('theano version: {:s}'.format(theano.__version__))
    print('pescador version: {:s}'.format(pescador.__version__))
    print('numpy version: {:s}'.format(np.__version__))

    root_folder = '/scratch/js7561/datasets/MedleyDB_output'
    model_base_folder = os.path.join(root_folder, 'models')
    # splitfile = '/home/js7561/dev/miasma/data/dataSplits_7_1_2.pkl'
    splitfile = '/home/js7561/dev/miasma/data/dataSplits_6_2_2.pkl'

    # Create a folder for this experiment
    model_folder = os.path.join(model_base_folder, expid)
    # Add random time delay to prevent concurrency crashes
    time.sleep(np.random.rand() * 10)
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    # for pool_layer in ['max', 'mean', 'softmax']:
    for pool_layer in pool_layers:

        print('\n------------- MODEL: {:s}-pooling -------------'.format(
            pool_layer))

        # Add random time delay (0-10 sec) to ensure parallel jobs don't
        # crash here
        time.sleep(np.random.rand() * 10)
        smp_folder = os.path.join(model_folder, pool_layer)
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
            'temp_conv': temp_conv,
            'freq_conv': freq_conv,
            'dropout': dropout,
            'reg_W': reg_W,
            'reg_activity': reg_activity,
            'pool_layer': pool_layer,
            'theano_version': theano.__version__,
            'keras_version:': keras.__version__,
            'numpy_version': np.__version__,
            'pescador_version': pescador.__version__}

        # print(metadata)
        json.dump(metadata, open(metadata_file, 'w'), indent=2)

        # Repeat for 5 train/validate/test splits
        for split_n, split_idx in enumerate(split_indices):

            print('\n---------- Split {:d} ----------'.format(split_idx))

            if split_n == 0:
                print_model_summary = True
            else:
                print_model_summary = False

            # Build model
            model = build_model(
                tf_rows=tf_rows, tf_cols=tf_cols, nb_filters=nb_filters,
                kernel_sizes=kernel_sizes,
                nb_fullheight_filters=nb_fullheight_filters, loss=loss,
                optimizer=optimizer, metrics=metrics, pool_layer=pool_layer,
                print_model_summary=print_model_summary, temp_conv=temp_conv,
                freq_conv=freq_conv, min_active_frames=min_active_frames,
                dropout=dropout, reg_W=reg_W, reg_activity=reg_activity)

            # Load data
            # if pool_layer == 'none':
            #     (train_generator, X_val, Y_val, ID_val,
            #      X_test, Y_test, ID_test) = (
            #         get_vad_data_frames(
            #             splitfile=splitfile,
            #             split_index=split_idx,
            #             root_folder=root_folder,
            #             augmentations=augs,
            #             feature='cqt44100_1024_8_36',
            #             activation='vocal_activation44100_1024',
            #             n_bag_frames=n_bag_frames,
            #             act_threshold=act_threshold,
            #             n_hop_frames=n_hop_frames,
            #             batch_size=batch_size,
            #             n_samples=n_samples,
            #             n_active=n_active,
            #             train_id=False,
            #             val_id=True,
            #             test_id=True))
            #
            # else:
            #     (train_generator, X_val, Y_val, ID_val,
            #      X_test, Y_test, ID_test) = (
            #         get_vad_data(
            #             splitfile=splitfile,
            #             split_index=split_idx,
            #             root_folder=root_folder,
            #             augmentations=augs,
            #             feature='cqt44100_1024_8_36',
            #             activation='vocal_activation44100_1024',
            #             n_bag_frames=n_bag_frames,
            #             min_active_frames=min_active_frames,
            #             act_threshold=act_threshold,
            #             n_hop_frames=n_hop_frames,
            #             batch_size=batch_size,
            #             n_samples=n_samples,
            #             n_active=n_active,
            #             train_id=False,
            #             val_id=True,
            #             test_id=True))

            # New code
            (train_generator, X_val, Y_val, ID_val,
             X_test, Y_test, ID_test) = (
                get_vad_data(
                    splitfile=splitfile,
                    split_index=split_idx,
                    root_folder=root_folder,
                    augmentations=augs,
                    feature='cqt44100_1024_8_36',
                    activation='vocal_activation44100_1024',
                    n_bag_frames=n_bag_frames,
                    min_active_frames=min_active_frames,
                    act_threshold=act_threshold,
                    n_hop_frames=n_hop_frames,
                    batch_size=batch_size,
                    n_samples=n_samples,
                    n_active=n_active,
                    train_id=False,
                    val_id=True,
                    test_id=True,
                    frame_level=(pool_layer == 'none')))

            checkpoint_file = os.path.join(
                smp_folder, 'weights_best{:d}.hdf5'.format(split_idx))

            # Train
            history = fit_model(model, checkpoint_file, train_generator, X_val,
                                Y_val, samples_per_epoch=samples_per_epoch,
                                nb_epochs=nb_epochs, verbose=verbose)

            pred = model.predict(X_test)

            acc = accuracy_score(
                Y_test.reshape(-1), 1 * (pred.reshape(-1) >= 0.5))
            print('Test accuracy: {:.3f}'.format(acc))

            # Save model
            modeljsonfile = os.path.join(
                smp_folder, 'model{:d}.json'.format(split_idx))
            model_json = model.to_json()
            with open(modeljsonfile, 'w') as json_file:
                json.dump(model_json, json_file, indent=2)

            # Save last version of weights (for resuming training)
            weights_last_file = os.path.join(
                smp_folder, 'weights_last{:d}.hdf5'.format(split_idx))
            model.save_weights(weights_last_file)

            # Save Y_test, predictions and IDs
            ytestfile = os.path.join(
                smp_folder, 'ytest{:d}.npy.gz'.format(split_idx))
            yprobfile = os.path.join(
                smp_folder, 'yprob{:d}.npy.gz'.format(split_idx))
            yidfile = os.path.join(
                smp_folder, 'yid{:d}.npy.gz'.format(split_idx))

            Y_test.dump(gzip.open(ytestfile, 'wb'))
            pred.dump(gzip.open(yprobfile, 'wb'))
            ID_test.dump(gzip.open(yidfile, 'wb'))

            # Save history
            history_score_file = os.path.join(
                smp_folder, 'history_scores{:d}.json'.format(split_idx))
            json.dump(history.history, open(history_score_file, 'w'), indent=2)

            history_file = os.path.join(
                smp_folder, 'history{:d}.pkl'.format(split_idx))
            pickle.dump(history, open(history_file, 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('expid', type=str)
    parser.add_argument('--n_bag_frames', type=int, default=44)
    parser.add_argument('--min_active_frames', type=int, default=10)
    parser.add_argument('--act_threshold', type=float, default=0.5)
    parser.add_argument('--n_hop_frames', type=int, default=22)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--n_active', type=int, default=128)
    parser.add_argument('--samples_per_epoch', type=int, default=1024)
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--tf_rows', type=int, default=288)
    parser.add_argument('--tf_cols', type=int, default=44)
    parser.add_argument('--nb_filters', type=int, nargs='+', default=[32, 16])
    parser.add_argument('--kernel_sizes', type=int, nargs='+',
                        default=[3, 3, 3, 3])
    parser.add_argument('--nb_fullheight_filters', type=int, default=16)
    parser.add_argument('--loss', type=str, default='binary_crossentropy')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['accuracy', 'precision', 'recall'])
    parser.add_argument('--split_indices', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4])
    parser.add_argument('--pool_layers', type=str, nargs='+',
                        default=['max', 'mean', 'softmax'])
    parser.add_argument('--temp_conv', type=int, default=0)
    parser.add_argument('--freq_conv', type=int, default=0)
    parser.add_argument('--augs', type=str, nargs='+', default=['original'])
    parser.add_argument('--dropout', action='store_const', const=True,
                        default=False)
    parser.add_argument('--reg_W', action='store_const', const=True,
                        default=False)
    parser.add_argument('--reg_activity', action='store_const', const=True,
                        default=False)

    args = parser.parse_args()

    # Convert kernel_sizes into list of tuple
    ind = 0
    kernel_sizes = []
    while ind < len(args.kernel_sizes):
        t = (args.kernel_sizes[ind], args.kernel_sizes[ind+1])
        kernel_sizes.append(t)
        ind += 2

    temp_conv = bool(args.temp_conv)
    freq_conv = bool(args.freq_conv)

    run_experiment(args.expid,
                   n_bag_frames=args.n_bag_frames,
                   min_active_frames=args.min_active_frames,
                   act_threshold=args.act_threshold,
                   n_hop_frames=args.n_hop_frames,
                   batch_size=args.batch_size,
                   n_samples=args.n_samples,
                   n_active=args.n_active,
                   samples_per_epoch=args.samples_per_epoch,
                   nb_epochs=args.nb_epochs,
                   verbose=args.verbose,
                   tf_rows=args.tf_rows,
                   tf_cols=args.tf_cols,
                   nb_filters=args.nb_filters,
                   kernel_sizes=kernel_sizes,
                   nb_fullheight_filters=args.nb_fullheight_filters,
                   loss=args.loss, optimizer=args.optimizer,
                   metrics=args.metrics,
                   split_indices=args.split_indices,
                   pool_layers=args.pool_layers,
                   temp_conv=temp_conv,
                   freq_conv=freq_conv,
                   augs=args.augs,
                   dropout=args.dropout,
                   reg_W=args.reg_W,
                   reg_activity=args.reg_activity)

