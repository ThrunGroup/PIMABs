# Exps

Contains code for running and analyzing a large number of experiments. This subdir is not fully documented yet.

## Usage

- To generate the experiment configuration a list of experiments to run, modify `create_config.py` to describe the experiments you want to create, and then run `python create_config.py`.
- To then run the experiments, run `python run.py -e auto_exp_config.py`.
- To parse the logs and view the scaling graphs, run `python parse_logs.py`.

## Description of files

- `create_config.py`:
- `auto_exp_config.py`: The list of experiments created by running `create_config.py` and which will be passed to `run.py`
- `parse_logs.py`:
- `run.py`:
- `snakevizcode.py`: