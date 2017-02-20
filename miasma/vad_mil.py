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




