#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
##  Adding debug to the parameters below will let you submit job to msuchard_develop.q i.e. node g502
#$ -l VEGA20,gpu,highp,h_rt=167:00:00,vega=1
#$ -l h=!g500
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Email address to notify
#$ -M kos@ucla.edu
# Notify when
#$ -m bea

 
# load the job environment:
. /u/local/Modules/default/init/modules.sh

export GPU_DEVICE_ORDINAL=$SGE_HGR_vega
echo $HOSTNAME
echo $GPU_DEVICE_ORDINAL
module load julia/1.9
module load amd/rocm
julia --project=/u/home/k/kose/ hawkes_model1_cb.jl ${JOB_ID} 100000
