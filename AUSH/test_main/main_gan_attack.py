# -*- coding: utf-8 -*-
# @Time       : 2019/8/24 11:08
# @Author     : chensi
# @File       : main_gan_attack.py
# @Software   : PyCharm
# @Desciption : None

import sys

sys.path.append("../")
import os, argparse
from utils.load_data.load_data import load_data
from model.attack_model.gan_attack.trainer import Train_GAN_Attacker
from utils.load_data.load_attack_info import load_attack_info
from utils.attack.data_to_file import *
import numpy as np


# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def gan_attack(data_set_name, attack_method, target_id, is_train, write_to_file=1, final_attack_setting=None):
    # 路径设置
    path_train = '../data/data/' + data_set_name + '_train.dat'
    path_test = '../data/data/' + data_set_name + '_test.dat'
    attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                        "../data/data/" + data_set_name + "_target_users"]
    model_path = "../result/model_ckpt/" + '_'.join([data_set_name, attack_method, str(target_id)]) + ".ckpt"

    # 读取seletced items和target users
    attack_info = load_attack_info(*attack_info_path)
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)
    # 攻击设置
    if len(attack_method.split('_')[1:]) == 2:
        attack_num, filler_num = map(int, attack_method.split('_')[1:])
        filler_method = 0
    else:
        attack_num, filler_num, filler_method = map(int, attack_method.split('_')[1:])
    selected_items = attack_info[target_id][0]

    #
    gan_attacker = Train_GAN_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                      selected_id_list=selected_items,
                                      filler_num=filler_num, attack_num=attack_num, filler_method=filler_method)
    #
    # if is_train:
    #     # 训练->模型保存->生成fake_profiles
    #     fake_profiles = gan_attacker.execute(is_train=True, model_path=model_path,
    #                                          final_attack_setting=final_attack_setting)
    # else:
    #     # restore>模型保存->生成fake_profiles
    #     fake_profiles, real_profiles = gan_attacker.execute(is_train=False, model_path=model_path,
    #                                                         final_attack_setting=final_attack_setting)
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
    # 数据集名称，用来选择训练数据路径
    parser.add_argument('--dataset', type=str, default='ml100k', help='filmTrust/ml100k/grocery')
    # 目标item
    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 5,395,181,565,254,601,623,619,64,558
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    parser.add_argument('--target_ids', type=str, default='62,1077,785,1419,1257,1319,1612,1509,1545,1373',
                        help='attack target list')
    # 参数 - 攻击数量，即往数据集里插入多少假用户
    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')
    # 参数 - filler数量，可理解为是每个假用户有多少评分
    parser.add_argument('--filler_num', type=int, default=90,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')
    # 参数 - 选择filler item的方法，0是随机
    parser.add_argument('--filler_method', type=str, default='', help='0/1/2/3')
    # 生成的攻击结果写入文件还是返回numpy矩阵，这里设置为1就好
    parser.add_argument('--write_to_file', type=int, default=1, help='write to fake profile to file or return array')
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
    attack_method = '_'.join(['gan', str(args.attack_num), str(args.filler_num), str(args.filler_method)]).strip('_')

    #
    for target_id in args.target_ids:
        """读取生成攻击时的sample的filler"""
        attackSetting_path = '_'.join(map(str, [args.dataset, args.attack_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        real_profiles, filler_indicator = np.load(attackSetting_path + '.npy')
        final_attack_setting = [args.attack_num, real_profiles, filler_indicator]

        """训练模型并注入攻击"""
        _ = gan_attack(args.dataset, attack_method, target_id, is_train,
                       write_to_file=args.write_to_file,
                       final_attack_setting=final_attack_setting)
