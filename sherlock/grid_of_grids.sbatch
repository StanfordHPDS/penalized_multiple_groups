#!/usr/bin/bash 
#SBATCH --job-name=test 
#SBATCH --time=2:00:00
#SBATCH -p normal
#SBATCH --error=test_job.%j.err
#SBATCH -c 1 
#SBATCH --mem=1GB
for i in 40; do
  for j in 0 1; do
    for k in 0 1 2; do
      for l in {1..100}; do
        sbatch indiv_batch.sbatch "$i" "$j" "$k" "$l"
      done
    done
  done
done