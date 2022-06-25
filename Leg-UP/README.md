
# Shilling Black-box Recommender Systems by Learning to Generate Fake User Profiles

This repository contains our implementation for Leg-UP (<ins>Le</ins>arning to <ins>G</ins>enerate Fake <ins>U</ins>ser <ins>P</ins>rofiles) and various shilling attack methods including AIA, DCGAN, WGAN, Random Attack, Average Attack, Segment Attack and Bandwagon Attack. 

Please kindly cite our paper [[IEEE Xplore](https://ieeexplore.ieee.org/document/9806457)] [[arXiv Preprint](https://arxiv.org/abs/2206.11433)] if you use it:

> Chen Lin, Si Chen, Meifang Zeng, Sheng Zhang, Min Gao, and Hui Li. 2022. Shilling Black-Box Recommender Systems by Learning to Generate Fake User Profiles. In TNNLS.

    @article{LinCZZGL22,
	  author    = {Chen Lin and
	               Si Chen and
	               Meifang Zeng and
	               Sheng Zhang and
	               Min Gao and
	               Hui Li},
	  title     = {Shilling Black-Box Recommender Systems by Learning to Generate Fake User Profiles},
	  journal   = {{IEEE} Trans. Neural Networks Learn. Syst.},
	  year      = {2022}
	}

## Environment
- Python 3.8
- higher 0.2.1
- scikit-learn 0.24.1
- scikit-surprise 1.1.1
- tensorflow 2.7
- pytorch 1.10
- numpy 1.20.1

## Data

The datasets used in our experiments can be found in the [data](../data) folder.


## Command Line Parameters
`run.py` is the main entry of the program, it requires several parameters:

- `data_set`: the recommendation dataset used in the experiment (Possible values: "ml100k", ''filmTrust'', ''automotive'', "yelp", ''GroceryFood'', ''ToolHome'' and ''AppAndroid''.  Default is  "ml100k").
- `attack_num`: number of injected profiles, i.e., A value (Default is 50).
- `filler_num`: number of fillers, i.e., P value (Default is 36).
- `surrogate`: surrogate RS model (Possible values: "WMF", ''ItemAE'', ''SVDpp'', and ''PMF''.  Default is  "WMF").
- `target_ids`: id of the target item (Default is 62).
- `recommender`: victim recommender (Possible values: ''AUSHplus'',  ''AIA'', ''WGANAttacker'', ''DCGANAttacker'', ''RandomAttacker'', ''AverageAttacker'', ''BandwagonAttacker'', and ''SegmentAttacker''.  Default is  "WMF"). Note that ''AUSHplus'' is the name of Leg-UP in our implementation.
- `cuda_id`: GPU id (Default is 0).
- `use_cuda`: use CPU or GPU (Default is 1).

## Examples

Please refer to `run.sh` for some running examples.



