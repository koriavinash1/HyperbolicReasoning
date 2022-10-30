#!/bin/bash
# Example of running python script in a batch mode
#SBATCH -c 4                        # Number of CPU Cores
#SBATCH -p gpus                     # Partition (queue)
#SBATCH --gres gpu:1                # gpu:n, where n = number of GPUs
#SBATCH --mem 12288                 # memory pool for all cores
#SBATCH --nodelist monal03               # SLURM node
#SBATCH --output=slurm.%N.%j.log    # Standard output and error log

# source environment
source /vol/biomedic3/as217/medicaltransformers/bin/activate

# Run python script
python /vol/biomedic3/as217/SymbolicInterpretability/main.py
# call script
nohup codebook.sh
