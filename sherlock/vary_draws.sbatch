#!/usr/bin/bash 
#SBATCH --job-name=test 
#SBATCH --time=2:00:00
#SBATCH -p normal
#SBATCH --error=test_job.%j.err
#SBATCH -c 1 
#SBATCH --mem=1GB
for i in 10 20 30 40 50 60 70 80 90 100; do
  for j in 0; do
    for k in 0 1 2; do
      for l in {1..25}; do
        sbatch indiv_batch.sbatch "$i" "$j" "$k" "$l"
      done
    done
  done
done