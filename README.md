# Attentive Session-Based Recommendations with Side Information

This repository contains a list of materials associated with a final project
for CS 249: Advanced Data Mining w/ Yizhou Sun, Spring 2021 at UCLA.
This repository can be used to replicate our project, when used in conjunction
with our final paper.

The members of our group are:
Christian Loanzon, Hamlin Liu, Michael Potter, and Yash Lala.

## Environment Setup

Our project's codebase is primarily in Python.

You can set it up using Conda, a Python environment manager.
For instructions on how to set up Conda, please see the Conda project's
[install guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Once Conda is installed, please run the following commands in the project's
root directory via your shell.

```
conda create --name cs-249-project --file requirements.txt

conda activate cs-249-project
```

This will create and activate a Conda environment with all the dependencies
required to use our project code. Note that the Conda environment is not
persistent -- when running our code in a new shell session, you will have to
re-run `conda activate cs-249-project`.

From here, you can look through the other files in our project.

All files with the suffix `.ipynb` can be run via Jupyter Notebook --
to open them, run `jupyter notebook` command in the root directory of the
project. This will open an interactive menu where you can view and run all
of our project code.

To run files with the `.py` suffix, please run `python3
FILE_I_WANT_TO_RUN.py` from your shell. Alternatively, you can view and
execute these `.py` files using Jupyter Notebook, as described above.

## Files

### Notebooks and Helper Scripts

- `data-scraping.ipynb`: This notebook contains a script that can be used to
  scrape plot summaries en-masse from the IMDb website. Instructions for its
  use are contained within.
- `GRU4REC-Gridsearch.ipynb`: This notebook can be used to search through
  hyperparameters for our regular GRU-based recommender model (ie. the
  non-attentive model).
- `GRU4RECAttention.ipynb`: This notebook contains our implementation of an
  attentive GRU4REC model. The included model leverages side features
  (ie. the IMDb plot summaries encoded by BERT).
- `GRU4RECF_notebook.ipynb`: This notebook contains our reference
  implementation of non-attentive GRU4REC. It does support alternating
  optimizers, loss function tweaks, and plot embeddings.
- NextItNet_run.py`: This python module contains an implementation of
  the NextItNet recommender system framework. When run as a script, it
  will train a recommender system on our data.
- NextItNet_GridSearch.ipynb`: This notebook contains everything in
- `GRU4RECF_Gridsearch.py`: This notebook contains the beginnings of
  a hyperparameter search for our regular GRU-based recommender model.
  It is not complete -- please use GRU4REC-Gridsearch.ipynb instead.
- `GRU4RECF_run.py`: This Python script contains a copied version of the
  GRU4RECF model described above. It also implements a wrapper script that runs
  the model on the given dataset.
- `NextItNet_run.py`, along with some extra code to search through
  hyperparameters en masse.
- `model.py`: This Python module contains our model's classes.
- `metrics.py`: This Python module contains the metric functions that we
  used to train and evaluate our design (eg. BPR loss).
- `preprocessing.py`: This Python module contains functions used to
  preprocess our dataset.
- `dataset.py`: This python module contains wrapper classes for our
  dataset.
- `utils.py`: This model contains some helper functions of use when
  manipulating data. For example, it allows us to convert BERT
  language embeddings from CSV form into dictionary form.
- `README.md`: Hopefully, you've figured out what this one does by now.

## Project Replication

### Data Scraping

To replicate our dataset from scratch, you should first download the `MovieLens
1M` and `MovieLens 20M` datasets from the
[GroupLens site](https://grouplens.org/datasets/movielens/).
While we only tested our model on the 1M dataset, you will need the 20M dataset
downloaded to generate the plot-augmented 1M dataset.

After downloading and extracting the data archive as described on the GroupLens
website, please place the MovieLens 1M data in `data/movielens-1m/`, and the 20M
data in `data/movielens-20m/`.
Now, you will want to augment the MovieLens 1M dataset with plot summaries to
use for our plot embeddings. You can do so by running the `data-scraping.ipynb`
notebook; instructions for its use are within. Note that this scraping process
involves manually visiting IMDb and copying in movie plots (some movies are
unable to be scraped directly from IMDb).
If you want to skip this lengthy process, we have included the MovieLens-1M
dataset with concatenated plots in the `data` directory of this repository.
The files in `data/` exactly mirror the MovieLens 1M dataset, with one
exception -- `movies.csv` now has an added "plot" column describing the plot of
the movie.

We have linked our generated DistilBERT plot embeddings below.
If the links are broken, please file an issue in this repository.

1. MovieLens 1M DistilBERT plot embeddings: save this file as
   `data/bert_sequence_1m.txt`. The data can be found
   [here](https://drive.google.com/file/d/1D6_jvNBs6YNhYU8Bd9UOKBVQrnoNAO78/view?usp=sharing).
2. MovieLens 20M DistilBERT plot embeddings: save this file as
   `data/bert_sequence_20m.txt`. The data can be found
   [here](https://drive.google.com/file/d/1zzXhfT4LrQIxDCWHlA34D-g1VVl2ZU7s/view?usp=sharing).

After the plot embeddings have been generated, we have enough data to run our
models.

### Results Replication

All Jupyter Notebooks here are self contained. They use the `preprocessing.py`
Python module (included here) to structure the data into their preferred forms.
You can run the Jupyter notebooks to directly obtain results.
