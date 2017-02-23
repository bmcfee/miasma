# CREATED: 2/23/17 17:11 by Justin Salamon <justin.salamon@nyu.edu>

import os
import gzip
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def eval_exp(expid):

    root_folder = '/scratch/js7561/datasets/MedleyDB_output'
    model_base_folder = os.path.join(root_folder, 'models')
    model_folder = os.path.join(model_base_folder, expid)

    split_indices = [2, 3, 4, 5, 6]

    # BAG-LEVEL EVAL
    print('BAG LEVEL EVALUATION')
    for pool_layer in ['softmax', 'max', 'mean']:

        print('\t{:s} POOLING'.format(pool_layer.capitalize()))
        smp_folder = os.path.join(model_folder, pool_layer)

        for split_n, split_idx in enumerate(split_indices):

            ytestfile = os.path.join(
                smp_folder, 'ytest{:d}.npy.gz'.format(split_idx))
            yprobfile = os.path.join(
                smp_folder, 'yprob{:d}.npy.gz'.format(split_idx))
            yidfile = os.path.join(
                smp_folder, 'yid{:d}.npy.gz'.format(split_idx))

            ytest = np.load(gzip.open(ytestfile, 'rb'))
            yprob = np.load(gzip.open(yprobfile, 'rb'))
            yid = np.load(gzip.open(yidfile, 'rb'))

            ypred_flat = 1 * (yprob.reshape(-1) >= 0.5)
            ytest_flat = ytest.reshape(-1)

            acc = accuracy_score(ytest_flat, ypred_flat)
            precision = precision_score(ytest_flat, ypred_flat)
            recall = recall_score(ytest_flat, ypred_flat)

            zeros = np.sum(1 * (ytest_flat == 0))
            ones = np.sum(1 * (ytest_flat == 1))
            baseline = max(zeros, ones) / float(zeros + ones)

            report = ('split{:d}: acc: {:.2f}\tpre: {:.2f}\trec: {:.2f}' +
                      '\t0/1: {:d}/{:d}\tbaseline: {:.2f}').format(
                split_idx, acc, precision, recall, zeros, ones, baseline)

            print(report)

