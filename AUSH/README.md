

# Attacking Recommender Systems with Augmented User Profiles

This repository contains one shilling attack algorithm, AUSH, published in the following paper [[ACM Library](https://dl.acm.org/doi/10.1145/3340531.3411884)] [[arXiv Preprint](https://arxiv.org/abs/2005.08164)]:

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
### Step1: Pre-processing
Use `test_main\data_preprocess.py` to transform amazon 5-cores ratings to tuples `[userid, itemid, normalized float rating]`.

Update on Dec 9, 2021: We have released several recommendation datasets for testing shilling attacks including the three datasets used in our CIKM'20 paper. You can directly use files in the [data](/data) folder for experiments. Please copy the data folder to the folder of AUSH before execution.

### Step2: Initialize
Use `test_main\data_preprocess.py`
 - select attack target
 - select attack number (default fix 50)
 - select filler size
 - selected items and target users
 - settings for bandwagon attack

### Step3: Training and Evaluation

 - Train baseline attack models
 ```shell script
python main_baseline_attack.py --dataset filmTrust --attack_methods average,segment,random,bandwagon --targets 601,623,619,64,558 --filler_num 36 --bandwagon_selected 103,98,115 --sample_filler 1
```
 - Evaluate baseline attack models
 ```shell script
python main_train_rec.py --dataset filmTrust --attack_method segment --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - RS performance before attack
 ```shell script
python main_train_rec.py --dataset filmTrust --attack_method no --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - Train AUSH
 ```shell script
python main_gan_attack.py --dataset filmTrust --target_ids 601,623,619,64,558 --filler_num 36
````

 - Evaluate AUSH
 ```shell script
python main_train_rec.py --dataset filmTrust --attack_method gan --model_name NMF_25 --target_ids 601,623,619,64,558 --filler_num 36
````

 - Comparative Study
 ```shell script
python main_eval_attack.py --dataset filmTrust --filler_num 36 --attack_methods gan,segment,average --rec_model_names NMF_25 --target_ids 601,623,619,64,558

python main_eval_similarity.py --dataset filmTrust --filler_num 36 --targets 601,623 --bandwagon_selected 103,98,115
```
