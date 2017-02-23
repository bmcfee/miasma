#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=02:00:00
#PBS -l mem=32gb
#PBS -N fpred
#PBS -M justin.salamon@gmail.com
#PBS -m abe
#PBS -j oe
#PBS -p 1000

module purge

SRCDIR=$HOME/dev/miasma

RUNDIR=$SCRATCH/jobs/miasma/run-${PBS_JOBID/.*}
mkdir $RUNDIR
cd $RUNDIR

module load cuda/7.5.18
module load cudnn/7.5v5.1
module load sox/intel/14.4.1

source activate py35hpcT9b

THEANO_FLAGS="base_compiledir=$PBS_JOBTMP,device=gpu,floatX=float32,mode=FAST_RUN" python $SRCDIR/miasma/vad_frame_predictions.py exp001