#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

module load python/3.10
module load libffi

echo "activating env"
source .venv/bin/activate

python src/bayes_gnn_opf.py