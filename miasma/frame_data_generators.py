# CREATED: 2/21/17 13:24 by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
import gzip
import os
from miasma.miasma.data_generators import batch_mux
import pescador
import glob


def _generate_bag_frames(trackid, cqt, act, frame, n_bag_frames,
                         act_threshold=0.5):

    # Compute bag label
    patch_act = act[frame:frame+n_bag_frames]

    # Compute bag ID
    bagid = '{:s}_{:d}'.format(trackid, frame)

    return dict(
        X=(cqt[:, frame:frame+n_bag_frames]).reshape(
            -1, cqt.shape[0], n_bag_frames, 1),
        Y=(1 * (patch_act >= act_threshold)).reshape(1, -1),
        ACT=patch_act.reshape(1, -1),
        ID=np.asarray([bagid]))


def mil_bag_generator_frames(cqtfile, actfile, n_bag_frames, n_hop_frames,
                             act_threshold=0.5, shuffle=True):
    # Load cqt file and ativation file
    cqt = np.load(gzip.open(cqtfile, 'rb'))
    act = np.load(gzip.open(actfile, 'rb'))

    # Get bag ID (from filename)
    trackid = '_'.join(os.path.basename(cqtfile).split('_')[:2])

    # librosa puts time in dim 1
    order = np.arange(0, cqt.shape[1] - n_bag_frames, n_hop_frames)
    if shuffle:
        np.random.shuffle(order)

    for frame in order:
        yield _generate_bag_frames(trackid, cqt, act, frame, n_bag_frames,
                                   act_threshold=act_threshold)


def infinite_mil_bag_generator_frames(cqtfile, actfile, n_bag_frames,
                                      n_hop_frames, act_threshold=0.5,
                                      shuffle=True):
    '''
    '''
    # Load cqt file and ativation file
    cqt = np.load(gzip.open(cqtfile, 'rb'))
    act = np.load(gzip.open(actfile, 'rb'))

    # Get bag ID (from filename)
    trackid = '_'.join(os.path.basename(cqtfile).split('_')[:2])

    frame = -1
    while True:
        if shuffle:
            frame = np.random.randint(0, cqt.shape[1] - n_bag_frames)
        else:
            frame = np.mod(frame + n_hop_frames, cqt.shape[1] - n_bag_frames)

        yield _generate_bag_frames(trackid, cqt, act, frame, n_bag_frames,
                                   act_threshold=act_threshold)


def vad_minibatch_generator_frames(
        root_folder, track_list, augmentations=['original'],
        feature='cqt44100_1024_8_36',
        activation='vocal_activation44100_1024', n_bag_frames=44,
        act_threshold=0.5, n_hop_frames=22, shuffle=True,
        batch_size=1, n_samples=None, n_active=1000, with_replacement=False):
    # Collect all feature and activation files
    cqt_files = []
    for track in track_list:
        for aug in augmentations:
            cqt_folder = os.path.join(root_folder, aug, 'features', feature)
            # cqtfile = os.path.join(cqt_folder, '{:s}_cqt.npy.gz'.format(track))
            files = (
                glob.glob(
                    os.path.join(cqt_folder, '{:s}*_cqt.npy.gz'.format(track))))
            for cqtf in files:
                cqt_files.append(cqtf)

    # Turn all files into streams
    streams = []
    for cqtfile in cqt_files:
        # get matching activation file
        actfolder = os.path.join(os.path.dirname(os.path.dirname(cqtfile)),
                                 activation)
        actfile = os.path.join(
            actfolder, os.path.basename(cqtfile).replace(
                '_cqt.npy.gz', '_vocalactivation.npy.gz'))
        assert os.path.isfile(actfile)

        if with_replacement:
            streams.append(
                pescador.Streamer(infinite_mil_bag_generator_frames, cqtfile,
                                  actfile, n_bag_frames, n_hop_frames,
                                  act_threshold, shuffle))
        else:
            streams.append(
                pescador.Streamer(mil_bag_generator_frames, cqtfile,
                                  actfile, n_bag_frames, n_hop_frames,
                                  act_threshold, shuffle))

    # Mux the streams into minimbatches
    batch_streamer = batch_mux(streams, batch_size, n_samples=n_samples,
                               n_active=n_active,
                               with_replacement=with_replacement)

    return batch_streamer


def keras_vad_minibatch_generator_frames(
        root_folder, track_list,
        augmentations=['original'],
        feature='cqt44100_1024_8_36',
        activation='vocal_activation44100_1024',
        n_bag_frames=44,
        act_threshold=0.5,
        n_hop_frames=22,
        shuffle=True,
        batch_size=1,
        n_samples=None,
        n_active=1000,
        with_replacement=False,
        with_id=False):
    '''
    '''
    keras_generator = vad_minibatch_generator_frames(
        root_folder, track_list, augmentations, feature, activation,
        n_bag_frames, act_threshold, n_hop_frames,
        shuffle, batch_size, n_samples, n_active, with_replacement)

    if with_id:
        for batch in keras_generator.generate():
            yield(batch['X'], batch['Y'], batch['ID'])
    else:
        for batch in keras_generator.generate():
            yield (batch['X'], batch['Y'])


def get_vad_data_generators_frames(
        splitfile='../data/dataSplits_7_1_2.pkl',
        split_index=2,
        root_folder='/scratch/js7561/datasets/MedleyDB_output/',
        augmentations=['original'],
        feature='cqt44100_1024_8_36',
        activation='vocal_activation44100_1024',
        n_bag_frames=44,
        act_threshold=0.5,
        n_hop_frames=22,
        batch_size=32,
        n_samples=None,
        n_active=1000,
        train_id=False,
        val_id=True,
        test_id=True):

    # Load data split
    split = np.load(splitfile)

    # TRAIN GENERATOR
    track_list = split[split_index][0]
    shuffle = True
    with_replacement = True

    train_generator = keras_vad_minibatch_generator_frames(
        root_folder, track_list, augmentations, feature, activation,
        n_bag_frames, act_threshold, n_hop_frames,
        shuffle, batch_size, n_samples, n_active, with_replacement,
        with_id=train_id)

    # VALIDATE GENERATOR
    track_list = split[split_index][1]
    shuffle = False
    with_replacement = False
    val_batch_size = batch_size

    validate_generator = keras_vad_minibatch_generator_frames(
        root_folder, track_list, ['original'], feature, activation,
        n_bag_frames, act_threshold, n_hop_frames,
        shuffle, val_batch_size, n_samples, n_active, with_replacement,
        with_id=val_id)

    # TEST GENERATOR
    track_list = split[split_index][2]
    shuffle = False
    with_replacement = False
    test_batch_size = batch_size

    test_generator = keras_vad_minibatch_generator_frames(
        root_folder, track_list, ['original'], feature, activation,
        n_bag_frames, act_threshold, n_hop_frames,
        shuffle, test_batch_size, n_samples, n_active, with_replacement,
        with_id=test_id)

    return train_generator, validate_generator, test_generator


def get_vad_data_frames(
        splitfile='../data/dataSplits_7_1_2.pkl',
        split_index=2,
        root_folder='/scratch/js7561/datasets/MedleyDB_output/',
        augmentations=['original'],
        feature='cqt44100_1024_8_36',
        activation='vocal_activation44100_1024',
        n_bag_frames=44,
        act_threshold=0.5,
        n_hop_frames=22,
        batch_size=32,
        n_samples=None,
        n_active=1000,
        train_id=False,
        val_id=True,
        test_id=True):

    train_generator, validate_generator, test_generator = (
        get_vad_data_generators_frames(
            splitfile=splitfile,
            split_index=split_index,
            root_folder=root_folder,
            augmentations=augmentations,
            feature=feature,
            activation=activation,
            n_bag_frames=n_bag_frames,
            act_threshold=act_threshold,
            n_hop_frames=n_hop_frames,
            batch_size=batch_size,
            n_samples=n_samples,
            n_active=n_active,
            train_id=train_id,
            val_id=val_id,
            test_id=test_id))

    # Get full validation & test data
    X_val = []
    Y_val = []
    ID_val = []

    for batch in validate_generator:
        X_val.extend(batch[0])
        Y_val.extend(batch[1])
        if val_id:
            ID_val.extend(batch[2])

    X_val = np.asarray(X_val)
    Y_val = np.asarray(Y_val)
    ID_val = np.asarray(ID_val)

    print('Validation set:')
    print(X_val.shape)
    print(Y_val.shape)
    print('0: {:d}'.format(np.sum(Y_val.reshape(-1) == 0)))
    print('1: {:d}'.format(np.sum(Y_val.reshape(-1) == 1)))
    print(' ')

    X_test = []
    Y_test = []
    ID_test = []

    for batch in test_generator:
        X_test.extend(batch[0])
        Y_test.extend(batch[1])
        if test_id:
            ID_test.extend(batch[2])

    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    ID_test = np.asarray(ID_test)

    print('Test set:')
    print(X_test.shape)
    print(Y_test.shape)
    print('0: {:d}'.format(np.sum(Y_test == 0)))
    print('1: {:d}'.format(np.sum(Y_test == 1)))
    print(' ')

    return train_generator, X_val, Y_val, ID_val, X_test, Y_test, ID_test

