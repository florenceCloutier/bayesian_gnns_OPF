#!/bin/bash

#SBATCH --time=50:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

module load python/3.10
module load libffi

echo "activating env"
source .venv/bin/activate

python src/canos_opf.py
