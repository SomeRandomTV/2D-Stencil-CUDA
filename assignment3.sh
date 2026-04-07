#!/bin/bash

interactive -p gpu-build

module purge
module load cuda

nvcc ./stencil.cu  -o stencil

./stencil input-stencil.pgm output.pgm

exit


