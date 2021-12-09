# Shilling Black-box Recommender Systems by Learning to Generate Fake User Profiles

This repository contains our implementation for Leg-UP (<ins>Le</ins>arning to <ins>G</ins>enerate Fake <ins>U</ins>ser <ins>P</ins>rofiles) and various shilling attack methods including AIA, DCGAN, WGAN, Random Attack, Average Attack, Segment Attack and Bandwagon Attack. 

More information will be provided when the paper is accepted.

## Environment
- Python 3.8
- higher 0.2.1
- scikit-learn 0.24.1
- scikit-surprise 1.1.1
- tensorflow 2.7
- pytorch 1.10
- numpy 1.20.1

## Data

The datasets used in our experiments can be found in the [Data](../Data) folder.

## Usage

Please refer to `run.sh` for usage.


## Command Line Parameters
`run.py` is the main entry of the program, it requires several parameters:

- `data_set`: the recommendation dataset used in the experiment (Possible values: "ml100k", '''XXX'.  Default is  "ml100k").
- `filler_num`: 

