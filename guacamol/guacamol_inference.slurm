#!/bin/bash
#SBATCH --job-name=guacamol_inference
#SBATCH --output ./guacamol_logs/out/guacamol_%j.txt
#SBATCH --error ./guacamol_logs/error/guacamol_%j.txt 
#SBATCH -p gpu_p    
#SBATCH --qos gpu_normal
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100_32gb
#SBATCH --nice=10000





apptainer exec --nv --overlay /lustre/groups/aih/jointformer/retmol_container/retmold_overlay.img /lustre/groups/aih/jointformer/retmol_container/retmol_old.sif helper_guacamol.sh
