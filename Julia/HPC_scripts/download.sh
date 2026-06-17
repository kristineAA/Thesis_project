#!/bin/bash
FILE="results_scalars_7scen_2026-06-17_12-03-50.csv"
# get the value after the last slash
FILE_NAME=$(basename $FILE)
# Upload to HPC
scp s204608@login.hpc.dtu.dk:Project/Master_Thesis/$FILE ./$FILE