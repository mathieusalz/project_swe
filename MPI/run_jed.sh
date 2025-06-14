#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --mem=10G

module purge
module load gcc openmpi hdf5


start_time=$(date +%s.%N)
srun ./swe
end_time=$(date +%s.%N)

elapsed=$(echo "$end_time - $start_time" | bc -l)
printf "Simulation completed in %.3f seconds\n" $elapsed
