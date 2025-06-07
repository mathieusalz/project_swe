#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --mem=5G

module purge
module load gcc hdf5


start_time=$(date +%s.%N)
srun ./swe
end_time=$(date +%s.%N)

elapsed=$(echo "$end_time - $start_time" | bc -l)
printf "Simulation completed in %.3f seconds\n" $elapsed
