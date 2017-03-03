# CREATED: 2/20/17 17:46 by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
import gzip
import os
import pescador
import glob
import time


def _bag_activation(activation, min_active_frames, act_threshold=0.5):
    '''
    Return bag-level label from activation curve.

    Given activation curve (time series) with values in the range [0, 1],
    the minimum number of required active frames and a threshold for considering
    a frame active (default 0.5), return whether the bag defined by this activation
    curve is active (returns 1) or not (returns 0).

    Parameters
    ----------
    activation : np.ndarray
        Activation curve
    min_active_frames : int
        Minimum number of contiguous active frames required to consider bag positive.
    act_threshold : float
        Threshold for considering single frame active (default = 0.5).

    Returns
    -------
    bag_label : int
        The bag label, which can be 0 (inactive) or 1 (active).
    '''
    condition = activation >= act_threshold

    # The following computes the length of every consecutive sequence
    # of True's in condition:
    active_lengths = np.diff(np.where(np.concatenate(
        ([condition[0]], condition[:-1] != condition[1:], [True])))[0])[::2]

    # Need at least min_active_frames to consider bag as positive
    if len(active_lengths) > 0:
        bag_label = 1 * (active_lengths.max() >= min_active_frames)
    else:
        bag_label = 0

    return bag_label


def _generate_bag(trackid, cqt, act, frame, n_bag_frames, min_active_frames,
                  act_threshold=0.5):
    '''
    Create VAD bag from full-length CQT and activation curve.

    Parameters
    ----------
    trackid : str
    cqt : np.ndarray
    act : np.ndarray
    frame : int
    n_bag_frames : int
    min_active_frames : int
    act_threshold : int

    Returns
    -------

    '''
    # Compute bag label
    patch_act = act[frame:frame + n_bag_frames]
    bag_label = _bag_activation(patch_act, min_active_frames,
                                act_threshold=act_threshold)

    # Compute bag ID
    bagid = '{:s}_{:d}'.format(trackid, frame)

    return dict(
        X=(cqt[:, frame:frame + n_bag_frames]).reshape(
            -1, cqt.shape[0], n_bag_frames, 1),
        Y=np.asarray([bag_label], dtype=np.int32),
        ID=np.asarray([bagid]))


def _generate_bag_frames(trackid, cqt, act, frame, n_bag_frames,
                         min_active_frames, act_threshold=0.5):

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


def mil_bag_generator(cqtfile, actfile, n_bag_frames, min_active_frames,
                      n_hop_frames, act_threshold=0.5, shuffle=True,
                      infinite=False, frame_level=False):
    '''
    Returns a in/finite MIL bag generator.

    The generator yields a dictionary with three elements:
    X = features, Y = label, Z = bag ID (trackid + first frame index).

    Parameters
    ----------
    cqtfile : str
        Path to .npy.gz file containing the log-CQT matrix
    actfile : str
        Path to .npy.gz file containing the activation vector
    n_bag_frames : int
        Number of frames to include in a bag
    min_active_frames: int
        Minimum number of consecutive active frames to consider bag positive
    n_hop_frames : int
        Number of frames to jump between consecutive bags. Ignored if
        (infinite=True and shuffle=True).
    shuffle : bool
        Whether to shuffle the ordering of the bags (for training) or not (for
        validation and test).
    infinite : bool
        If True, generates infinitely many samples. If False, generates
        samples until frames are exhausted.

    Returns
    -------
    bag : dictionary with X = features, Y = label, Z = bag ID
    '''
    # Load cqt file and ativation file
    cqt = np.load(gzip.open(cqtfile, 'rb'))
    act = np.load(gzip.open(actfile, 'rb'))

    # Get bag ID (from filename)
    trackid = '_'.join(os.path.basename(cqtfile).split('_')[:2])

    if frame_level:
        gen_func = _generate_bag_frames
    else:
        gen_func = _generate_bag

    if infinite:
        frame = -1
        while True:
            if shuffle:
                frame = np.random.randint(0, cqt.shape[1] - n_bag_frames)
            else:
                frame = np.mod(frame + n_hop_frames,
                               cqt.shape[1] - n_bag_frames)

            yield gen_func(trackid, cqt, act, frame, n_bag_frames,
                           min_active_frames, act_threshold=act_threshold)
    else:
        # librosa puts time in dim 1
        order = np.arange(0, cqt.shape[1]-n_bag_frames, n_hop_frames)
        if shuffle:
            np.random.shuffle(order)

        for frame in order:
            yield gen_func(trackid, cqt, act, frame, n_bag_frames,
                           min_active_frames, act_threshold=act_threshold)


def batch_mux(streams, batch_size, n_samples=None, n_active=1000,
              with_replacement=False, revive=False, lam=32.0):
    '''
    Multiplex streams into batches of size n_batch

    Parameters
    ----------
    streams : list of pescador.Streamer
        The list of streams to multiplex
    batch_size : int > 0
        Number of samples to in each batch (batch size)
    n_samples : int or None
        Number of individual samples to generate (limit). If None, generate
        infinite number of samples (unless with_replacement is False in which
        case generate until all streams are exhausted)
    n_active : int > 0
        Number of streams that can be active simultaneously
    with_replacement : bool
        If true sample form streams indefinitely. If False streams are sampled
        until exhausted.

    Returns
    -------
    batch_streamer : pescador.Streamer
        Batch multiplexer
    '''

    stream_mux = pescador.Streamer(
        pescador.mux, streams, n_samples, n_active,
        with_replacement=with_replacement, revive=revive, lam=lam)

    batch_streamer = pescador.Streamer(
        pescador.buffer_streamer, stream_mux, batch_size)

    return batch_streamer


def vad_minibatch_generator(root_folder, track_list,
                            augmentations=['original'],
                            feature='cqt44100_1024_8_36',
                            activation='vocal_activation44100_1024',
                            n_bag_frames=44,
                            min_active_frames=2,
                            act_threshold=0.5,
                            n_hop_frames=22,
                            shuffle=True,
                            batch_size=100,
                            n_samples=None,
                            n_active=1000,
                            with_replacement=False,
                            revive=False,
                            infinite=False,
                            lam=32.0,
                            frame_level=False):
    '''
    Returns a minibatch generator for VAD (yields in pescador format).

    Parameters
    ----------
    root_folder
    track_list
    augmentations
    feature
    activation
    n_bag_frames
    min_active_frames
    act_threshold
    n_hop_frames
    shuffle
    batch_size
    n_samples
    n_active
    with_replacement

    Returns
    -------

    '''
    # Collect all feature files
    cqt_files = []
    for track in track_list:
        for aug in augmentations:
            cqt_folder = os.path.join(root_folder, aug, 'features', feature)
            # cqtfile = os.path.join(cqt_folder, '{:s}_cqt.npy.gz'.format(track))
            files = (
                glob.glob(os.path.join(
                    cqt_folder, '{:s}_*_cqt.npy.gz'.format(track))))
            for cqtf in files:
                cqt_files.append(cqtf)

    # DEBUG
    # raise ValueError('found {:d} files'.format(len(cqt_files)))
    print("Data generator found {:d} files".format(len(cqt_files)))
    # print(cqt_files)

    # Turn all files into streams
    streams = []
    for cqtfile in cqt_files:
        # get matching activation file
        actfolder = os.path.join(
            os.path.dirname(os.path.dirname(cqtfile)), activation)
        actfile = os.path.join(
            actfolder, os.path.basename(cqtfile).replace(
                '_cqt.npy.gz', '_vocalactivation.npy.gz'))
        assert os.path.isfile(actfile)

        streams.append(
            pescador.Streamer(mil_bag_generator, cqtfile, actfile,
                              n_bag_frames, min_active_frames, n_hop_frames,
                              act_threshold=act_threshold,
                              shuffle=shuffle,
                              infinite=infinite,
                              frame_level=frame_level))

    # DEBUG
    #     print("Done")

    # Mux the streams into minimbatches
    batch_streamer = batch_mux(streams, batch_size,
                               n_samples=n_samples,
                               n_active=n_active,
                               with_replacement=with_replacement,
                               revive=revive,
                               lam=lam)

    return batch_streamer


def keras_vad_minibatch_generator(root_folder, track_list,
                                  augmentations=['original'],
                                  feature='cqt44100_1024_8_36',
                                  activation='vocal_activation44100_1024',
                                  n_bag_frames=44,
                                  min_active_frames=10,
                                  act_threshold=0.5,
                                  n_hop_frames=22,
                                  shuffle=True,
                                  batch_size=32,
                                  n_samples=None,
                                  n_active=1000,
                                  with_replacement=False,
                                  revive=False,
                                  infinite=False,
                                  lam=32.0,
                                  with_id=False,
                                  frame_level=False):
    '''
    Returns a minibatch generator for VAD (yields in keras format).

    Parameters
    ----------
    root_folder
    track_list
    augmentations
    feature
    activation
    n_bag_frames
    min_active_frames
    act_threshold
    n_hop_frames
    shuffle
    batch_size
    n_samples
    n_active
    with_replacement

    Returns
    -------

    '''
    keras_generator = vad_minibatch_generator(
        root_folder, track_list,
        augmentations=augmentations, feature=feature, activation=activation,
        n_bag_frames=n_bag_frames, min_active_frames=min_active_frames,
        act_threshold=act_threshold, n_hop_frames=n_hop_frames,
        shuffle=shuffle, batch_size=batch_size, n_samples=n_samples,
        n_active=n_active, with_replacement=with_replacement, revive=revive,
        infinite=infinite, lam=lam, frame_level=frame_level)

    if with_id:
        for batch in keras_generator.generate():
            yield (batch['X'], batch['Y'], batch['ID'])
    else:
        for batch in keras_generator.generate():
            yield (batch['X'], batch['Y'])


def get_vad_data_generators(
        splitfile='../data/dataSplits_7_1_2.pkl',
        split_index=2,
        root_folder='/scratch/js7561/datasets/MedleyDB_output/',
        augmentations=['original'],
        feature='cqt44100_1024_8_36',
        activation='vocal_activation44100_1024',
        n_bag_frames=44,
        min_active_frames=10,
        act_threshold=0.5,
        n_hop_frames=22,
        batch_size=32,
        n_samples=None,
        n_active=128,
        train_id=False,
        val_id=True,
        test_id=True,
        frame_level=False):

    # Load data split
    split = np.load(splitfile)

    # TRAIN GENERATOR
    track_list = split[split_index][0]
    with_replacement = False
    shuffle = True
    revive = True
    infinite = True
    lam = 32.0
    
    # print('train generator:')
    train_generator = keras_vad_minibatch_generator(
        root_folder, track_list, augmentations=augmentations, feature=feature,
        activation=activation, n_bag_frames=n_bag_frames,
        min_active_frames=min_active_frames, act_threshold=act_threshold,
        n_hop_frames=n_hop_frames, shuffle=shuffle, batch_size=batch_size,
        n_samples=n_samples, n_active=n_active,
        with_replacement=with_replacement, revive=revive, infinite=infinite,
        lam=lam, with_id=train_id, frame_level=frame_level)

    # VALIDATE GENERATOR
    track_list = split[split_index][1]
    with_replacement = False
    shuffle = False
    revive = False
    infinite = False
    lam = None
    val_n_active = 1
    val_batch_size = batch_size

    # print('validate generator:')
    validate_generator = keras_vad_minibatch_generator(
        root_folder, track_list, augmentations=['original'], feature=feature,
        activation=activation, n_bag_frames=n_bag_frames,
        min_active_frames=min_active_frames, act_threshold=act_threshold,
        n_hop_frames=n_hop_frames, shuffle=shuffle, batch_size=val_batch_size,
        n_samples=n_samples, n_active=val_n_active,
        with_replacement=with_replacement, revive=revive, infinite=infinite,
        lam=lam, with_id=val_id, frame_level=frame_level)

    # TEST GENERATOR
    # print('test generator:')
    track_list = split[split_index][2]
    with_replacement = False
    shuffle = False
    revive = False
    infinite = False
    lam = None
    test_n_active = 1
    test_batch_size = batch_size

    test_generator = keras_vad_minibatch_generator(
        root_folder, track_list, augmentations=['original'], feature=feature,
        activation=activation, n_bag_frames=n_bag_frames,
        min_active_frames=min_active_frames, act_threshold=act_threshold,
        n_hop_frames=n_hop_frames, shuffle=shuffle, batch_size=test_batch_size,
        n_samples=n_samples, n_active=test_n_active,
        with_replacement=with_replacement, revive=revive, infinite=infinite,
        lam=lam, with_id=test_id, frame_level=frame_level)

    return train_generator, validate_generator, test_generator


def get_vad_data(
        splitfile='../data/dataSplits_7_1_2.pkl',
        split_index=2,
        root_folder='/scratch/js7561/datasets/MedleyDB_output/',
        augmentations=['original'],
        feature='cqt44100_1024_8_36',
        activation='vocal_activation44100_1024',
        n_bag_frames=44,
        min_active_frames=10,
        act_threshold=0.5,
        n_hop_frames=22,
        batch_size=32,
        n_samples=None,
        n_active=128,
        train_id=False,
        val_id=True,
        test_id=True,
        frame_level=False):

    train_generator, validate_generator, test_generator = (
        get_vad_data_generators(
            splitfile=splitfile,
            split_index=split_index,
            root_folder=root_folder,
            augmentations=augmentations,
            feature=feature,
            activation=activation,
            n_bag_frames=n_bag_frames,
            min_active_frames=min_active_frames,
            act_threshold=act_threshold,
            n_hop_frames=n_hop_frames,
            batch_size=batch_size,
            n_samples=n_samples,
            n_active=n_active,
            train_id=train_id,
            val_id=val_id,
            test_id=test_id,
            frame_level=frame_level))

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
    print('0: {:d}'.format(np.sum(1 * (Y_val.reshape(-1) == 0))))
    print('1: {:d}'.format(np.sum(1 * (Y_val.reshape(-1) == 1))))
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

