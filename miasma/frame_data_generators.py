# CREATED: 2/21/17 13:24 by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np
import gzip
import os
from miasma.miasma.data_generators import batch_mux
import pescador


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
            cqtfile = os.path.join(cqt_folder, '{:s}_cqt.npy.gz'.format(track))
            cqt_files.append(cqtfile)

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
        with_replacement=False):
    '''
    '''
    keras_generator = vad_minibatch_generator_frames(
        root_folder, track_list, augmentations, feature, activation,
        n_bag_frames, act_threshold, n_hop_frames,
        shuffle, batch_size, n_samples, n_active, with_replacement)

    for batch in keras_generator.generate():
        yield (batch['X'].reshape(-1), batch['Y'].reeshape(-1))

