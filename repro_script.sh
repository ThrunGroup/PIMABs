#!/bin/bash

git submodule update --remote
pip install algorithms/napg
pip install -e .
python repro_script_python.py