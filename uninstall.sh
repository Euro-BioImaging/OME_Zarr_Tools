#!/usr/bin/env bash

unalias fiji
rm -rf ~/Applications;
rm -rf ~/OME_Zarr_Tools
rm -rf ~/miniconda3

echo "export PATH=/opt/munge-0.5.12/bin:/opt/slurm-21.08.8/bin:/opt/slurm-21.08.8/sbin:/opt/nhc-1.4.2/sbin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin" > ~/.bashrc
source  ~/.bashrc
