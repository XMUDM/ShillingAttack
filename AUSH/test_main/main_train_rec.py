# -*- coding: utf-8 -*-
# @Time       : 2019/8/23 19:29
# @Author     : chensi
# @File       : main_train_rec.py
# @Software   : PyCharm
# @Desciption : None
import sys, os, argparse

sys.path.append("../")
from utils.load_data.load_data import *
from model.trainer_rec import rec_trainer
from model.trainer_rec_surprise import basic_rec
from utils.attack.data_to_file import target_prediction_writer


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train_rec(data_set_name, model_name, attack_method, target_id, is_train):
    if attack_method == "no":
        attack_method = ""
        model_path = "../result/model_ckpt/" + '_'.join([model_name, data_set_name]) + ".ckpt"
    else:
        model_path = "../result/model_ckpt/" + '_'.join([model_name, data_set_name, attack_method]) + ".ckpt"
    path_train = "../data/data_attacked/" + '_'.join([data_set_name, str(target_id), attack_method]) + ".dat"
    path_test = "../data/data/" + data_set_name + "_test.dat"
    if attack_method == "": path_train = "../data/data/" + data_set_name + "_train.dat"

    # load_data
    dataset_class = load_data(path_train=path_train, path_test=path_test,
                              header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)
    # train rec
    if model_name in ["IAutoRec", "UAutoRec", "NNMF"]:
        predictions, hit_ratios = rec_trainer(model_name, dataset_class, target_id, is_train, model_path)
    else:
        predictions, hit_ratios = basic_rec(model_name, path_train, path_test, target_id)

    # write to file
    dst_path = "../result/pred_result/" + '_'.join([model_name, data_set_name, str(target_id), attack_method])
    dst_path = dst_path.strip('_')
    target_prediction_writer(predictions, hit_ratios, dst_path)


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='automotive', help='input data_set_name,filmTrust or ml100k')

    parser.add_argument('--model_name', type=str, default='NMF_25', help='NNMF,IAutoRec,UAutoRec,NMF_25')

    parser.add_argument('--attack_method', type=str, default='G1',
                        help='no,gan,segment,average,random,bandwagon')

    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 5,395,181,565,254,601,623,619,64,558
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    parser.add_argument('--target_ids', type=str, default='866',
                        help='attack target')

    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')

    parser.add_argument('--filler_num', type=int, default=4,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')

    args = parser.parse_args()
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()

    """train"""
    if args.attack_method == 'no':
        attack_method_ = args.attack_method
    else:
        attack_method_ = '_'.join([args.attack_method, str(args.attack_num), str(args.filler_num)])
    is_train = 1
    train_rec(args.dataset, args.model_name, attack_method_, args.target_ids[0], is_train=is_train)
    for target in args.target_ids[1:]:
        if args.attack_method == 'no':
            is_train = 0
        train_rec(args.dataset, args.model_name, attack_method_, target, is_train=is_train)
