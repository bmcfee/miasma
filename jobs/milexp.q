#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=03:00:00
#PBS -l mem=16gb
#PBS -N mil
#PBS -M justin.salamon@gmail.com
#PBS -m abe
#PBS -j oe
#PBS -p 0

module purge

SRCDIR=$HOME/dev/miasma

RUNDIR=$SCRATCH/jobs/miasma/run-${PBS_JOBID/.*}
mkdir $RUNDIR
cd $RUNDIR

module load cuda/7.5.18
module load cudnn/7.5v5.1
module load sox/intel/14.4.1

source activate py35hpcT9b

THEANO_FLAGS="base_compiledir=$PBS_JOBTMP,device=gpu,floatX=float32,mode=FAST_RUN" python $SRCDIR/miasma/vad_mil.py exp001 --n_bag_frames 44 --min_active_frames 10 --act_threshold 0.5 --n_hop_frames 22 --batch_size 32 --n_active 1000 --samples_per_epoch 1024 --nb_epochs 50 --verbose 1 --tf_rows 288 --tf_cols 44 --nb_filters 32 32 --kernel_sizes 3 3 3 3 --nb_fullheight_filters 32 --loss binary_crossentropy --optimizer adam --metrics accuracy precision recall --split_indices 2 --pool_layers softmax