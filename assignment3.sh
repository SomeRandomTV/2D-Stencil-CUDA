#!/bin/bash
#SBATCH -J pmpp-assignment3
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH -p gpu-build
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=2048MB

module purge
module load cuda

nvcc ./stencil.cu  -o stencil

./stencil input-stencil.pgm output.pgm




