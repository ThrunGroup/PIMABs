# Exps

Contains code for running and analyzing a large number of experiments. This subdir is not fully documented yet.

## Usage

- To generate the experiment configuration a list of experiments to run, modify `create_config.py` to describe the experiments you want to create, and then run `python create_config.py`.
- To then run the experiments, run `python run.py -e auto_exp_config.py`.
- To parse the logs and view the scaling graphs, run `python parse_logs.py`.

## Description of files

- `create_config.py`: Creates a list of experiments to be programmatically fed to `run.py`. Modify this to generate new experiments. 
- `auto_exp_config.py`: The list of experiments created by running `create_config.py` and which will be passed to `run.py`
- `parse_logs.py`: Analyzes the data from the experiments. To change which experiments you're analyzing, modify this file in the same way you modified `create_config.py`, namely with the parameters of the experiments you want to analyze.
- `run.py`: Runs the actual experiments. Pass a file with the list of experiments with the `-e` flag.
- `snakevizcode.py`: Helper code to parse the profile stats dumped by `cProfile.Profile()`'s `dump_stats` method. Might not be used any more.