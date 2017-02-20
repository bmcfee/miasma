# CREATED: 2/15/17 5:28 PM by Justin Salamon <justin.salamon@nyu.edu>

import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

import pescador
import gzip
import os
import fnmatch

np.random.seed(1337)  # for reproducibility


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


def _generate_bag(trackid, cqt, act, frame, n_bag_frames, min_active_frames, act_threshold=0.5):
    '''
    Create VAD bad from full-length CQT and activation curve.

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
    bag_label = _bag_activation(patch_act, min_active_frames, act_threshold=act_threshold)

    # Compute bag ID
    bagid = '{:s}_{:d}'.format(trackid, frame)

    return dict(
        #         X=patch,
        X=(cqt[:, frame:frame + n_bag_frames]).reshape(-1, cqt.shape[0], n_bag_frames, 1),
        Y=np.asarray([bag_label], dtype=np.int32),
        ID=np.asarray([bagid]))

