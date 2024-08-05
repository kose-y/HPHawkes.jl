#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
##  Adding debug to the parameters below will let you submit job to msuchard_develop.q i.e. node g502
#$ -l VEGA20,gpu,highp,h_rt=72:00:00,vega=1,h_data=10G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 4
# Email address to notify
#$ -M kos@ucla.edu
# Notify when
#$ -m bea

 
# load the job environment:
. /u/local/Modules/default/init/modules.sh

export GPU_DEVICE_ORDINAL=`tr ' ' ',' <<< $SGE_HGR_vega`
echo $GPU_DEVICE_ORDINAL
module load intel/2020.4
module load julia/1.9
module load amd/rocm
mpirun -np 4 julia --project=/u/home/k/kose/ hawkes_dist_model1_cb.jl ${JOB_ID}