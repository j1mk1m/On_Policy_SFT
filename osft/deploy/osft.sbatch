#!/bin/bash
#SBATCH --partition general
#SBATCH --time 1-23:00 # set time
#SBATCH --job-name=osft
#SBATCH --chdir=/home/gyeongwk/On_Policy_SFT/osft # working directory

#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8 # cpus

#SBATCH --output=logs/%x-%j.out # output log
#SBATCH --error=logs/%x-%j.err # error log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j1mk1m.dummy@gmail.com

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate osft

bash examples/osft_string_task.sh


