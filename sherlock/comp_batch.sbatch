#!/usr/bin/bash 
#SBATCH --job-name=test 
#SBATCH --time=2:00:00
#SBATCH -p normal
#SBATCH --error=test_job.%j.err
#SBATCH -c 1 
#SBATCH --mem=8GB
module load python/3.12.1
module load py-numpy/1.26.3_py312
module load py-pandas/2.2.1_py312
module load py-scikit-learn/1.5.1_py312
module load py-scipy/1.12.0_py312
python3 code/comps.py $1 $2