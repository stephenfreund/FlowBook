#!/bin/bash

# add slurm_logs to the front of ever arg in $*
args=()
for arg in $*; do
    args+=("slurm_logs/$arg")
done

data_ferret_timers ${args[@]}
