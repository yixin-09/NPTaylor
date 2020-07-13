#!/usr/bin/env bash
python apply_patch4v.py -n $1 -t $2
python run_patch4v.py -n $1 -t $2

