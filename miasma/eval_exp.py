# CREATED: 2/23/17 17:11 by Justin Salamon <justin.salamon@nyu.edu>

import os
import gzip
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


def training_report(model_folder, split_indices):

    print('\n-------------------- TRAINING REPORT --------------------')
    for pool_layer in ['softmax', 'max', 'mean', 'none']:
        print('\n{:s} POOLING'.format(pool_layer.upper()))
        smp_folder = os.path.join(model_folder, pool_layer)

        tm_all = []

        for split_n, split_idx in enumerate(split_indices):
            hscorefile = os.path.join(
                smp_folder, 'history_scores{:d}.json'.format(split_idx))
            tm = training_metrics(hscorefile)
            tm_all.append(tm)

        tm_all = np.asarray(tm_all)
        best_epoch = np.asarray([d['best_epoch'] for d in tm_all])
        best_acc = np.asarray([d['best_acc'] for d in tm_all])
        best_pre = np.asarray([d['best_pre'] for d in tm_all])
        best_rec = np.asarray([d['best_rec'] for d in tm_all])
        last_epoch = np.asarray([d['last_epoch'] for d in tm_all])
        last_acc = np.asarray([d['last_acc'] for d in tm_all])
        last_pre = np.asarray([d['last_pre'] for d in tm_all])
        last_rec = np.asarray([d['last_rec'] for d in tm_all])
        report = ('BEST (epochs {:d}/{:d}/{:d}/{:d}/{:d}) average: acc {:.2f}\t'
                  'pre {:.2f}\trec {:.2f}')
        report_tuple = tuple(best_epoch) + (best_acc.mean(), best_pre.mean(),
                                            best_rec.mean())
        report = report.format(*report_tuple)
        print(report)
        report = ('LAST (epochs {:d}/{:d}/{:d}/{:d}/{:d}) average: acc {:.2f}\t'
                  'pre {:.2f}\trec {:.2f}')
        report_tuple = tuple(last_epoch) + (last_acc.mean(), last_pre.mean(),
                                            last_rec.mean())
        report = report.format(*report_tuple)
        print(report)

    # Collect training curves
    acc_curves = {}
    loss_curves = {}
    for pool_layer in ['softmax', 'max', 'mean', 'none']:
        smp_folder = os.path.join(model_folder, pool_layer)
        for split_n, split_idx in enumerate(split_indices):
            hscorefile = os.path.join(
                smp_folder, 'history_scores{:d}.json'.format(split_idx))
            hscore = np.load(hscorefile)
            if pool_layer not in acc_curves.keys():
                acc_curves[pool_layer] = {'train': [], 'val': []}
                loss_curves[pool_layer] = {'train': [], 'val': []}
            acc_curves[pool_layer]['train'].append(hscore['acc'])
            acc_curves[pool_layer]['val'].append(hscore['val_acc'])
            loss_curves[pool_layer]['train'].append(hscore['acc'])
            loss_curves[pool_layer]['val'].append(hscore['val_acc'])

    # # epochs = np.arange(1, len(hscore['acc']) + 1)
    # epochs = np.arange(len(hscore['acc']))
    # # fig = plt.figure()
    # # ax1 = fig.add_subplot(221)
    #
    # for pool_layer in ['softmax', 'max', 'mean', 'none']:
    #     fig = plt.figure()
    #     plt.plot(epochs, acc_curves['train'], 'b')
    #     plt.plot(epochs, acc_curves['val'], 'r')
    #     plt.title('training accuracy')
    #     plt.ylabel('accuracy')
    #     plt.xlabel('epoch')
    #     plt.legend(['train', 'validate'], loc='upper left')
    #     plt.show()
    #
    # # # summarize history for loss
    # # ax2 = fig.add_subplot(222)
    # # plt.plot(epochs, history['loss'])
    # # plt.plot(epochs, history['val_loss'])
    # # plt.title('model loss')
    # # plt.ylabel('loss')
    # # plt.xlabel('epoch')
    # # plt.legend(['train', 'validate'], loc='upper left')
    # # plt.show()

    return acc_curves, loss_curves


def training_metrics(history_score_file):
    history = json.load(open(history_score_file, 'r'))

    best_epoch = np.argmin(history['val_loss'])
    best_acc = history['acc'][best_epoch]
    best_pre = history['precision'][best_epoch]
    best_rec = history['recall'][best_epoch]

    last_epoch = len(history['acc']) - 1
    last_acc = history['acc'][-1]
    last_pre = history['precision'][-1]
    last_rec = history['recall'][-1]

    return {'best_epoch': best_epoch,
            'best_acc': best_acc,
            'best_pre': best_pre,
            'best_rec': best_rec,
            'last_epoch': last_epoch,
            'last_acc': last_acc,
            'last_pre': last_pre,
            'last_rec': last_rec}


def eval_exp(expid):

    root_folder = '/scratch/js7561/datasets/MedleyDB_output'
    model_base_folder = os.path.join(root_folder, 'models')
    model_folder = os.path.join(model_base_folder, expid)

    split_indices = [2, 3, 4, 5, 6]

    # dataframe for storing ALL results
    df = pd.DataFrame(
        columns=['level', 'pooling', 'split', 'accuracy', 'precision',
                 'recall', 'zeros', 'ones', 'baseline'])

    # BAG-LEVEL EVAL
    print('-------------------- BAG LEVEL EVALUATION --------------------')
    for pool_layer in ['softmax', 'max', 'mean']:

        print('\n{:s} POOLING'.format(pool_layer.upper()))
        smp_folder = os.path.join(model_folder, pool_layer)

        split_results = []

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

            split_results.append([acc, precision, recall, zeros, ones,
                                  baseline])

            df.loc[len(df)] = (['bag', pool_layer, split_idx, acc, precision, 
                                recall, zeros, ones, baseline])

        # Average results
        split_results = np.asarray(split_results)
        avg_acc = split_results[:, 0].mean()
        avg_precision = split_results[:, 1].mean()
        avg_recall = split_results[:, 2].mean()
        avg_zeros = split_results[:, 3].mean()
        avg_ones = split_results[:, 4].mean()
        avg_baseline = split_results[:, 5].mean()

        report = ('averag: acc: {:.2f}\tpre: {:.2f}\trec: {:.2f}' +
                  '\t0/1: {:.0f}/{:.0f}\tbaseline: {:.2f}').format(
            avg_acc, avg_precision, avg_recall, avg_zeros, avg_ones,
            avg_baseline)
        print(colored(report, 'magenta', attrs=['bold']))

    # FRAME-LEVEL EVAL
    print('\n-------------------- FRAME LEVEL EVALUATION --------------------')

    for pool_layer in ['softmax', 'max', 'mean', 'none']:

        print('\n{:s} POOLING'.format(pool_layer.upper()))
        if pool_layer == 'none':
            frame_folder = os.path.join(model_folder, pool_layer)
        else:
            frame_folder = os.path.join(model_folder, pool_layer, 'frame_level')

        split_results = []

        for split_n, split_idx in enumerate(split_indices):

            ytestfile = os.path.join(
                frame_folder, 'ytest{:d}.npy.gz'.format(split_idx))
            yprobfile = os.path.join(
                frame_folder, 'yprob{:d}.npy.gz'.format(split_idx))
            yidfile = os.path.join(
                frame_folder, 'yid{:d}.npy.gz'.format(split_idx))

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

            split_results.append([acc, precision, recall, zeros, ones,
                                  baseline])

            df.loc[len(df)] = (['frame', pool_layer, split_idx, acc, 
                                precision, recall, zeros, ones, baseline])

        # Average results
        split_results = np.asarray(split_results)
        avg_acc = split_results[:, 0].mean()
        avg_precision = split_results[:, 1].mean()
        avg_recall = split_results[:, 2].mean()
        avg_zeros = split_results[:, 3].mean()
        avg_ones = split_results[:, 4].mean()
        avg_baseline = split_results[:, 5].mean()

        report = ('averag: acc: {:.2f}\tpre: {:.2f}\trec: {:.2f}' +
                  '\t0/1: {:.0f}/{:.0f}\tbaseline: {:.2f}').format(
            avg_acc, avg_precision, avg_recall, avg_zeros, avg_ones,
            avg_baseline)
        print(colored(report, 'magenta', attrs=['bold']))

    # PLOT BAG-LEVEL and FRAME-LEVEL BOX PLOTS
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca()
    df.boxplot(column=['accuracy'], by=['level', 'pooling'], ax=ax)
    plt.show()

    # Training report


    return df


