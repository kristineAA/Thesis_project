#!/bin/bash
#BSUB -J julia
#BSUB -q hpc
#BSUB -n 64
#BSUB -W 01:00
#BSUB -o julia_%J.out
#BSUB -e julia_%J.er

# Run the job
julia run.jl