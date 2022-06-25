# Shilling Attacks against Recommender Systems

This repository contains our implementations for Shilling Attacks against Recommender Systems. 

Folder structure:
- `AUSH`: The implementation of AUSH used in our CIKM'20 paper [[ACM Library](https://dl.acm.org/doi/10.1145/3340531.3411884)] [[arXiv Preprint](https://arxiv.org/abs/2005.08164)].
- `Leg-UP`: The implementation of Leg-UP in our TNNLS'22 paper [[IEEE Xplore](https://ieeexplore.ieee.org/document/9806457)] [[arXiv Preprint](https://arxiv.org/abs/2206.11433)] and a unified framework for comparing Leg-UP with various attackers including AIA, DCGAN, WGAN, Random Attack, Average Attack, Segment Attack and Bandwagon Attack.
- `data`: Recommendation datasets used in our experiments.

See `README.md` in each folder for more details.

Please kindly cite our papers if you find our implementations useful:

> Chen Lin, Si Chen, Hui Li, Yanghua Xiao, Lianyun Li, and Qian Yang. 2020. Attacking Recommender Systems with Augmented User Profiles. In CIKM. 855–864.

> Chen Lin, Si Chen, Meifang Zeng, Sheng Zhang, Min Gao, and Hui Li. 2022. Shilling Black-Box Recommender Systems by Learning to Generate Fake User Profiles. In TNNLS.

    @inproceedings{Lin2020Attacking,  
	  author    = {Chen Lin and
	               Si Chen and
	               Hui Li and
	               Yanghua Xiao and
	               Lianyun Li and
	               Qian Yang},
	  title     = {Attacking Recommender Systems with Augmented User Profiles},
	  booktitle = {{CIKM}},
	  pages     = {855--864},
	  year      = {2020}
    }  
    

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