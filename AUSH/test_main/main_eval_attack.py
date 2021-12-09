# -*- coding: utf-8 -*-
# @Time       : 2019/8/24 10:05
# @Author     : chensi
# @File       : main_eval_attack.py
# @Software   : PyCharm
# @Desciption : None
import sys, argparse
import numpy as np
import pandas as pd

sys.path.append("../")
from utils.load_data.load_data import load_data
from utils.load_data.load_attack_info import *


def attack_evaluate(real_preds_path, attacked_preds_file, non_rated_users, target_users):
    #
    names = ['uid', 'rating', 'HR_1', 'HR_3', 'HR_5', 'HR_10', 'HR_20', 'HR_50']
    real_preds = pd.read_csv(real_preds_path, sep='\t', names=names, engine='python')
    attacked_preds = pd.read_csv(attacked_preds_file, sep='\t', names=names, engine='python')
    # pred
    shift_target = np.mean(attacked_preds.iloc[target_users, 1].values - real_preds.iloc[target_users, 1].values)
    shift_all = np.mean(attacked_preds.iloc[non_rated_users, 1].values - real_preds.iloc[non_rated_users, 1].values)
    #
    HR_real_target = real_preds.iloc[target_users, range(2, 8)].mean().values
    HR_real_all = real_preds.iloc[non_rated_users, range(2, 8)].mean().values

    HR_attacked_target = attacked_preds.iloc[target_users, range(2, 8)].mean().values
    HR_attacked_all = attacked_preds.iloc[non_rated_users, range(2, 8)].mean().values
    return shift_target, HR_real_target, HR_attacked_target, shift_all, HR_real_all, HR_attacked_all


def eval_attack(data_set_name, rec_model_name, attack_method, target_id):
    dir = "../result/pred_result/"
    real_preds_path = dir + '_'.join([rec_model_name, data_set_name, str(target_id)])
    attacked_preds_file = real_preds_path + "_" + attack_method
    """
    ml100k
    """
    if data_set_name == 'ml100k':
        path_train = "../data/data/ml100k_train.dat"
        path_test = "../data/data/ml100k_test.dat"
        attack_info_path = ["../data/data/ml100k_selected_items", "../data/data/ml100k_target_users"]
    elif data_set_name == 'filmTrust':
        path_train = "../data/data/filmTrust_train.dat"
        path_test = "../data/data/filmTrust_test.dat"
        attack_info_path = ["../data/data/filmTrust_selected_items", "../data/data/filmTrust_target_users"]

    else:
        path_train = "../data/data/" + data_set_name + "_train.dat"
        path_test = "../data/data/" + data_set_name + "_test.dat"
        attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                            "../data/data/" + data_set_name + "_target_users"]

    attack_info = load_attack_info(*attack_info_path)
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)

    #
    target_users = attack_info[target_id][1]
    non_rated_users = dataset_class.get_item_nonrated_users(target_id)
    #
    res = attack_evaluate(real_preds_path, attacked_preds_file, non_rated_users, target_users)
    #
    target, all = res[:3], res[3:]
    target_str = '\t'.join([str(target[0]), '\t'.join(map(str, target[1])), '\t'.join(map(str, target[2]))])
    all_str = '\t'.join([str(all[0]), '\t'.join(map(str, all[1])), '\t'.join(map(str, all[2]))])

    # info
    info = '\t'.join([rec_model_name, attack_method, str(target_id)])
    # print(info + '\t' + target_str + '\t' + all_str)
    return info + '\t' + target_str + '\t' + all_str


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='automotive', help='filmTrust/ml100k/office')

    parser.add_argument('--attack_num', type=int, default=50, help='50 for ml100k and filmTrust')

    parser.add_argument('--filler_num', type=int, default=4, help='90 for ml100k,36 for filmTrust')

    parser.add_argument('--attack_methods', type=str, default='G0,G1',
                        help='gan,G0,G1,segment,average,random,bandwagon')

    parser.add_argument('--rec_model_names', type=str, default='NNMF,IAutoRec,UAutoRec,NMF_25',
                        help='NNMF,IAutoRec,UAutoRec,NMF_25')

    # filmTrust:5,395,181,565,254,601,623,619,64,558 - random*5+tail*5
    # ml100k:62,1077,785,1419,1257,1319,1612,1509,1545,1373 - random*5+tail*5
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    # 88,22,122,339,1431,1141,1656,477,1089,866
    parser.add_argument('--target_ids', type=str, default='88,22,122,339,1431,1141,1656,477,1089,866',
                        help='target_id')

    #
    args = parser.parse_args()
    #
    args.attack_methods = args.attack_methods.split(',')
    args.rec_model_names = args.rec_model_names.split(',')
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()
    """eval"""
    result = []

    for attack_method in args.attack_methods:
        for rec_model_name in args.rec_model_names:
            for target_id in args.target_ids:
                attack_method_ = '_'.join([attack_method, str(args.attack_num), str(args.filler_num)])
                try:
                    result_ = eval_attack(args.dataset, rec_model_name, attack_method_, target_id)
                    result.append(result_.split('\t'))
                except:
                    print(attack_method, rec_model_name, target_id)

    result = np.array(result).transpose()
    result = pd.DataFrame(dict(zip(range(result.shape[0]), result)))
    result.to_excel(args.dataset + '_performance_all.xls', index=False)
