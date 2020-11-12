# Paper

This is the code repository for AUSH: a GAN-based RS attacking framework  

Please kindly cite our paper:

@inproceedings{Lin2020Attacking,
author = {Lin, Chen and Chen, Si and Li, Hui and Xiao, Yanghua and Li, Lianyun and Yang, Qian},
title = {Attacking Recommender Systems with Augmented User Profiles},
year = {2020},
booktitle = {Proceedings of the 29th ACM International Conference on Information & Knowledge Management},
pages = {855–864},
location = {Virtual Event, Ireland},
series = {CIKM '20}
}


### step1:Pre-processing
test_main\data_preprocess.py transforms amazon 5cores ratings to tuples [userid,itemid, normalized float rating]

### step2: Initialize
test_main\data_preprocess.py
 - select attack target
 - select attack number (default fix 50)
 - select filler size
 - selected items and target users
 - settings for bandwagon attack

### step3. Training

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
 
 





