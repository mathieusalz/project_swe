#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --mem=10G

module purge
module load intel intel-oneapi-mpi intel-oneapi-vtune hdf5

vtune -collect hotspots -r prof_results -- ./swe