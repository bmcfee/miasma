# CREATED: 2/23/17 14:57 by Justin Salamon <justin.salamon@nyu.edu>

import argparse
import os
import shutil
import subprocess


def launch_prediction(expid):

    # Create folder for experiment jobs
    expfolder = os.path.expanduser(
        os.path.join('~/dev/miasma/jobs/frame_predictions', expid))
    if not os.path.isdir(expfolder):
        os.mkdir(expfolder)

    # Load in template
    qfile = os.path.expanduser(
        '~/dev/miasma/jobs/frame_predictions/framepred.q')
    with open(qfile, 'r') as file:
        data = file.readlines()
        file.close()

    # Get the last line (the one we care about)
    d = data[-1].split(' ')

    # for pooling_layer in ['softmax', 'max', 'mean', 'none']:
    #     for split_idx in [2, 3, 4, 5, 6]:

    # Copy qsub template
    # destfile = os.path.expanduser(
    #     os.path.join(expfolder, 'fpred_{:s}{:d}.q'.format(
    #         pooling_layer, split_idx)))
    destfile = os.path.expanduser(os.path.join(expfolder, 'fpred.q'))
    shutil.copyfile(qfile, destfile)

    data_fold = data[:]

    assert d[3][:3] == 'exp'
    d[3] = expid

    assert data_fold[4].startswith('#PBS -N ')
    # data_fold[4] = '#PBS -N fpred{:d}_{:s}_{:d}\n'.format(
    #     int(expid[3:]), pooling_layer, split_idx)
    data_fold[4] = '#PBS -N fpred{:d}\n'.format(int(expid[3:]))

    # add a couple of empty lines at the end
    data_fold.append('\n')
    data_fold.append('\n')

    with open(destfile, 'w') as file:
        file.writelines(data_fold)

    print(' ')
    print('Launching {:s}'.format(os.path.basename(destfile)))
    print(data_fold[-3])

    jobid = subprocess.check_output('qsub {:s}'.format(destfile),
                                    shell=True)
    jobid = jobid.strip('\n')
    print('jobid = {:s} ({:d})'.format(jobid, len(jobid)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('expid', type=str)

    args = parser.parse_args()

    launch_prediction(args.expid)

