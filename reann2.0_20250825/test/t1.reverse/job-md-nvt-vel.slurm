#!/bin/sh
#SBATCH -J vel-test
#SBATCH -p intel96
#SBATCH -N 1 -n 16
##SBATCH --gres=gpu:1
##SBATCH --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

source /public/home/xjf/.bash_profile
source /public/software/profile.d/mpi_intelmpi-2017.4.239.sh
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH #for specify the cuda path to overcome error: failed to open libnvrtc-builtins.so.11.1

##mem=`nvidia-smi -q -d Memory |grep -A4 GPU|grep Free|sed -e "s/Free.*://g"|sed -e "s/MiB//g"`
#mem=`nvidia-smi -q -d Memory |grep -A4 GPU|grep Used|sed -e "s/Used.*://g"|sed -e "s/MiB//g"`
#mem=`echo $mem`
#id_sel=`python -c "import sys;import numpy as np;arr_mem=list(map(float,sys.argv[1:]));id_sel=np.argmin(np.array(arr_mem));print(id_sel)" $mem`
#echo $mem
#echo $id_sel
#export CUDA_VISIBLE_DEVICES=$id_sel
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS=8

path="/public/home/xjf/code_tmp/REANN-github-0408-Z/lammps-libtorch-1.10/build-cpu/"
rm log-*
theme=nvt
log_out=out-$theme
date>$log_out
pwd>>$log_out
echo start `date +"%s"` >> $log_out
numproc=1
mpirun -n $numproc $path/lmp_mpi -in in-$theme.lmp -log log.lmp-$theme >>$log_out
echo end `date +"%s"` >> $log_out
