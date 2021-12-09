# -*- coding: utf-8 -*-
# @Time       : 2019/8/24 11:08
# @Author     : chensi
# @File       : main_gan_attack_baseline.py
# @Software   : PyCharm
# @Desciption : None

import sys

sys.path.append("../")
import os, argparse
from utils.load_data.load_data import load_data
from model.attack_model.gan_attack.trainer_baseline import Train_G_Attacker
from utils.load_data.load_attack_info import load_attack_info
from utils.attack.data_to_file import *
import numpy as np


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def gan_attack(data_set_name, attack_method, target_id, is_train, write_to_file=1, final_attack_setting=None):
    path_train = '../data/data/' + data_set_name + '_train.dat'
    path_test = '../data/data/' + data_set_name + '_test.dat'
    attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                        "../data/data/" + data_set_name + "_target_users"]

    attack_info = load_attack_info(*attack_info_path)
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)

    if len(attack_method.split('_')[1:]) == 2:
        attack_num, filler_num = map(int, attack_method.split('_')[1:])
        filler_method = 0
    else:
        attack_num, filler_num, filler_method = map(int, attack_method.split('_')[1:])

    loss_setting = int(attack_method.split('_')[0][-1])
    selected_items = attack_info[target_id][0]
    model_path = "../result/model_ckpt/" + '_'.join([data_set_name, attack_method, str(target_id)]) + ".ckpt"

    #
    gan_attacker = Train_G_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                    selected_id_list=selected_items,
                                    filler_num=filler_num, attack_num=attack_num, filler_method=filler_method,
                                    loss_setting=loss_setting)
    # if is_train:
    #     fake_profiles = gan_attacker.execute(is_train=True, model_path=model_path)
    # else:
    #     fake_profiles, real_profiles = gan_attacker.execute(is_train=False, model_path=model_path)
    #     if write_to_file == 0:
    #         return fake_profiles, real_profiles
    fake_profiles, real_profiles, filler_indicator = gan_attacker.execute(is_train=is_train, model_path=model_path,
                                                                          final_attack_setting=final_attack_setting)
    gan_attacker.sess.close()
    # """inject and write to file"""
    if write_to_file == 1:
        dst_path = "../data/data_attacked/" + '_'.join([data_set_name, str(target_id), attack_method]) + ".dat"
        attacked_file_writer(path_train, dst_path, fake_profiles, dataset_class.n_users)
    return fake_profiles, real_profiles, filler_indicator


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='automotive', help='filmTrust/ml100k/grocery')

    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 5,395,181,565,254,601,623,619,64,558
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    # 88,22,122,339,1431,1141,1656,477,1089,866
    parser.add_argument('--target_ids', type=str, default='88,22,122,339,1431,1141,1656,477,1089,866',
                        help='attack target list')

    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')

    parser.add_argument('--filler_num', type=int, default=4,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')

    parser.add_argument('--filler_method', type=str, default='', help='0/1/2/3')

    parser.add_argument('--write_to_file', type=int, default=1, help='write to fake profile to file or return array')

    parser.add_argument('--loss', type=int, default=1, help='0:reconstruction,1:reconstruction+seed')
    #
    args = parser.parse_args()
    #
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()
    """train"""
    is_train = 1
    attack_method = '_'.join(
        ['G' + str(args.loss), str(args.attack_num), str(args.filler_num), str(args.filler_method)]).strip('_')
    #
    for target_id in args.target_ids:

        attackSetting_path = '_'.join(map(str, [args.dataset, args.attack_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        real_profiles, filler_indicator = np.load(attackSetting_path + '.npy')
        final_attack_setting = [args.attack_num, real_profiles, filler_indicator]


        _ = gan_attack(args.dataset, attack_method, target_id, is_train,
                       write_to_file=args.write_to_file,
                       final_attack_setting=final_attack_setting)

    # gan_attack(args.dataset, attack_method, args.target_id, is_train, write_to_file=args.write_to_file)
