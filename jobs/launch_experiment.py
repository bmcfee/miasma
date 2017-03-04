# CREATED: 2/22/17 16:52 by Justin Salamon <justin.salamon@nyu.edu>

import argparse
import os
import shutil
import subprocess


def launch_experiment(expid):

    # Create folder for experiment jobs
    expfolder = os.path.expanduser(os.path.join('~/dev/miasma/jobs', expid))
    if not os.path.isdir(expfolder):
        os.mkdir(expfolder)

    # Load in template
    qfile = os.path.expanduser('~/dev/miasma/jobs/milexp.q')
    with open(qfile, 'r') as file:
        data = file.readlines()
        file.close()

    # Get the last line (the one we care about)
    d = data[-1].split(' ')

    # for pooling_layer in ['softmax']:
    #     for split_idx in [2]:
    for pooling_layer in ['softmax', 'max', 'mean', 'none']:
        for split_idx in [2, 3, 4, 5, 6]:

            # Copy qsub template
            destfile = os.path.expanduser(
                os.path.join(expfolder, 'milexp_{:s}{:d}.q'.format(
                    pooling_layer, split_idx)))
            shutil.copyfile(qfile, destfile)

            data_fold = data[:]

            assert d[3][:3] == 'exp'
            d[3] = expid

            assert d[-14] == '--split_indices'
            d[-13] = str(split_idx)

            assert d[-12] == '--pool_layers'
            d[-10] = pooling_layer

            data_fold[-1] = ' '.join(d)

            assert data_fold[4].startswith('#PBS -N ')
            data_fold[4] = '#PBS -N mil{:d}_{:s}_{:d}\n'.format(
                int(expid[3:]), pooling_layer, split_idx)

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

    launch_experiment(args.expid)

