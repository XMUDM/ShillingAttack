#!/bin/bash
 
for target_id in 5 395 181 565 254 601 623 619 64 558
do
	for rec_model_name in IAUtoRec UAUtoRec NNMF NMF_25
	do
		python main_eval_attack.py --dataset filmTrust --rec_model_name $rec_model_name --attack_method G0 --target_id $target_id --attack_num 50 --filler_num 36 >> filmTrust_result_G0
		#nohup python main_gan_attack_baseline.py --dataset filmTrust --target_id 5 --attack_num 50 --filler_num 36 --loss 0 >> G0_log 2>&1 &
	done
done