#!/bin/bash

env FERRET_PROFILE_CHECKPOINT=1 \
env FERRET_PROFILE_DIFF=1 \
data_ferret_slurm $*  -- execute_sdc
