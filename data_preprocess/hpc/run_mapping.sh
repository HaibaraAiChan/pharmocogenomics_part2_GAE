#!/bin/bash
export PYTHONPATH=/home/syang29/.conda/envs/py_3.5/lib/python3.5/site-packages:$PYTHONPATH
cd /work/syang29/py_3.5_multi/
python hpc_multi.py --infile $1  2>&1