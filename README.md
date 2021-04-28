# Algorithms for Shilling Attacks against Recommender Systems

This repository contains our implementations for Shilling Attacks against Recommender Systems. Currently, it includes one shilling attack algorithm, AUSH, published in the following paper [[Arxiv Preprint](https://arxiv.org/abs/2005.08164)]:

> Chen Lin, Si Chen, Hui Li, Yanghua Xiao, Lianyun Li, and Qian Yang. 2020. Attacking Recommender Systems with Augmented User Profiles. In CIKM. 855–864.

Please kindly cite our paper if you use it:

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

## How to run AUSH.
### step1: Pre-processing
test_main\data_preprocess.py transforms amazon 5cores ratings to tuples [userid,itemid, normalized float rating]

### step2: Initialize
test_main\data_preprocess.py
 - select attack target
 - select attack number (default fix 50)
 - select filler size
 - selected items and target users
 - settings for bandwagon attack

### step3: Training

 - baseline attack models
 ```shell script
python main_baseline_attack.py --dataset filmTrust --attack_methods average,segment,random,bandwagon --targets 601,623,619,64,558 --filler_num 36 --bandwagon_selected 103,98,115 --sample_filler 1
```
 - evaluation
 ```shell script
python main_train_rec.py --dataset filmTrust --attack_method segment --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - RS performance before attack
 ```shell script
python main_train_rec.py --dataset filmTrust --attack_method no --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - training AUSH
 ```shell script
python main_gan_attack.py --dataset filmTrust --target_ids 601,623,619,64,558 --filler_num 36
````

 - Evluation (AUSH)
 ```shell script
python main_train_rec.py --dataset filmTrust --attack_method gan --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - Comparative Study
 ```shell script
python main_eval_attack.py --dataset filmTrust --filler_num 36 --attack_methods gan,segment,average --rec_model_names NMF_25 --target_ids 601,623,619,64,558

python main_eval_similarity.py --dataset filmTrust --filler_num 36 --targets 601,623 --bandwagon_selected 103,98,115
```
 
 





