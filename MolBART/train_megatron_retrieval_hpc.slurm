#!/bin/bash
#SBATCH --job-name=guacamol_train
#SBATCH --output ./trainlogs/out/train__%j.txt
#SBATCH --error ./trainlogs/error/train_%j.txt 
#SBATCH -p gpu_p
#SBATCH --qos gpu_long 
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=32G
#SBATCH --constraint=v100_32gb
#SBATCH --nice=10000


apptainer exec --nv /lustre/groups/aih/jointformer/retmol_container/retmol_old.sif ./helper_train_megatron_retrieval_hpc.sh