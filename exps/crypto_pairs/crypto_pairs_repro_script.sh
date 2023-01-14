#!/bin/bash

pip install -e ../..
python ../../data/crypto_pairs/preprocess_crypto_pairs.py  # Make a crypto pairs dataset
python run_crypto_pairs_scaling.py  # Run the sfcaling experiment