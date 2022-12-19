# Faster Maximum Inner Product Search in High Dimensions

# Dear Reviewer:

We thank you for your time in reviewing our submission. 
We understand that you are performing a service to the community.
In order to ensure that review of this code is as easy as possible, we have:
- Included a one-line script to recreate all results (`repro_script.sh`)
- Heavily commented and documented the code
- Included an overview of the codebase and the files.

Thank you for reviewing our submission!

# Description of Files

The files are organized as follows:

- `algorithms/` contains the different MIPS algorithms: BanditMIPS is implemented in `action_elimination.py` and BanditMIPS-$\alpha$ is implemented in `adaptive_action_elimination.py`. The most recently-developed baseline algorithm, BoundedME, is implemented in `median_elimination.py`.
- `data/` contains the data used in the experiments. If you choose to reproduce the results, please download the data files into their respective folders and run `data/netflix/preproces_netflix.py` and `data/movie_lens/preprocess_movie_lens_1m.py`.
- `exps/` contains the scripts for the experiments.
  - the two main types of experiments are the scaling experimentsand the precision/speedup tradeoff experiments
- `utils/` contains helper functions used across the codebase.
- `logs/` contains the logs of the experiments.
- `figures/` contains the figures from analyzing the experimental results.

# Obtaining the datasets

- The Netflix Prize dataset is available from [here](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
- The Movie Lens 1m data is available from [here](https://grouplens.org/datasets/movielens/1m/).

# Installing HNSW
```bash
$ pip install algorithms/hnswlib
```

# Reproducing the results
- To reproduce all of the experimental logs and create the figures, run `chmod +x repro_script.sh && ./repro_script.sh`
