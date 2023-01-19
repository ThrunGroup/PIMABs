#!/bin/bash

git submodule update --remote
pip install -e .
pip install algorithms/hnswlib
python repro_script_python.py