#!/bin/bash
#SBATCH --gpus-per-node=1 # Number of GPUs
#SBATCH --mem=8192MB # Memory (8GB)
#SBATCH --time=20:25:00 # Max runtime
#SBATCH --partition=class # Partition
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --job-name=ssl-l1 # Job name
#SBATCH --output=l1-ep-100-250.%j.out # Output log file
# Load environment
source ~/.bashrc
# Activate your virtual environment
source /mnt/home/sattum/vit_covid_xai/vit_env/bin/activate
# Move to the directory from which sbatch was run
cd $SLURM_SUBMIT_DIR
# Run the Python script with dataset path
python /mnt/home/sattum/vit_covid_xai/vit_covid_train.py --data_dir /mnt/home/sattum/vit_covid_xai/data/
# Deactivate environment
deactivate
