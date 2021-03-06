#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:05:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course 
#SBATCH --gres=gpu:1

module purge

module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
module load matplotlib/2.1.1-foss-2017b-Python-3.6.3
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

cp -r $HOME/assignment_2/part2 $TMPDIR
cd $TMPDIR/part2
mkdir models

python3 train.py --batch_size 512 --sample_every 200 --print_every 200 --txt_file book_NL_darwin_reis_om_de_wereld.txt

cp -r models/* $HOME/assignment_2
cp results.txt $HOME/assignment_2