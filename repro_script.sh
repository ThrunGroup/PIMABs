#!/bin/bash

git submodule update --init --recursive
pip install -e .
pip install algorithms/napg
python repro_script_python.py