#!/usr/bin/bash 
#SBATCH --job-name=test 
#SBATCH --time=12:00:00
#SBATCH -p normal
#SBATCH --error=test_job.%j.err
#SBATCH -c 1 
#SBATCH --mem=8GB
module load python/3.12.1
module load py-pandas/2.2.1_py312
python3 result_combiner.py