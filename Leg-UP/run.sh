#!/bin/bash

#=================================================

for target_id in 62 785 1077 1257 1419; do
  python run.py --data_set ml100k --target_ids $target_id --attacker_list AUSHplus >log_ml100k_$target_id
done

for target_id in 5 395 181 565 254; do
  python run.py --data_set filmTrust --target_ids $target_id --attacker_list AUSHplus >log_filmTrust_$target_id
done

for target_id in 119 422 594 884 1593; do
  python run.py --data_set automotive --target_ids $target_id --attacker_list AUSHplus >log_automotive_$target_id
done
#=================================================

for attacker in AUSHplus AIA WGANAttacker DCGANAttacker RandomAttacker AverageAttacker BandwagonAttacker SegmentAttacker; do
  for target_id in 62 785 1077 1257 1419; do
    python run.py --data_set ml100k --target_ids $target_id --attacker_list $attacker >log_ml100k_$target_id"_"$attacker
  done

  for target_id in 5 395 181 565 254; do
    python run.py --data_set filmTrust --target_ids $target_id --attacker_list $attacker >log_filmTrust_$target_id"_"$attacker
  done

  for target_id in 119 422 594 884 1593; do
    python run.py --data_set automotive --target_ids $target_id --attacker_list $attacker >log_automotive_$target_id"_"$attacker
  done
done

#=================================================

for attacker in AUSHplus_SR AUSHplus_woD AUSHplus_SF AUSHplus_inseg; do
  for target_id in 62 785 1077 1257 1419; do
    python run.py --data_set ml100k --target_ids $target_id --attacker_list $attacker >log_ml100k_$target_id"_"$attacker
  done

  for target_id in 5 395 181 565 254; do
    python run.py --data_set filmTrust --target_ids $target_id --attacker_list $attacker >log_filmTrust_$target_id"_"$attacker
  done

  for target_id in 119 422 594 884 1593; do
    python run.py --data_set automotive --target_ids $target_id --attacker_list $attacker >log_automotive_$target_id"_"$attacker
  done
done

#=================================================
