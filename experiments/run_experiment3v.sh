#!/usr/bin/env bash
python apply_patch3v.py -n $1 -t $2
python run_patch3v.py -n $1 -t $2

