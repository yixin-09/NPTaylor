#!/usr/bin/env bash
python apply_patch2v.py -n $1 -t $2
python run_patch2v.py -n $1 -t $2

