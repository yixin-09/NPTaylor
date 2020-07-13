#!/usr/bin/env bash
python apply_patch1v.py -n $1 -t $2
python run_patch1v.py -n $1 -t $2

