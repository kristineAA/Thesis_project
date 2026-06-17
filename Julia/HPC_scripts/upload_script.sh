#!/bin/bash
FILE="job.sh"
# get the value after the last slash
FILE_NAME=$(basename $FILE)
# Upload to HPC
scp $FILE s204608@login.hpc.dtu.dk:Project/Master_Thesis/$FILE