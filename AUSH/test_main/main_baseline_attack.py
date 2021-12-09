# -*- coding: utf-8 -*-
# @Time       : 2019/8/23 11:49
# @Author     : chensi
# @File       : main_attack_baseline.py
# @Software   : PyCharm
# @Desciption : None

import sys, argparse

sys.path.append("../")
from utils.load_data.load_data import *
from utils.load_data.load_attack_info import *
from model.attack_model.baseline import *
from utils.attack.data_to_file import *
from model.attack_model.gan_attack.trainer import Train_GAN_Attacker


def get_data(data_set_name):
    path_train = '../data/data/' + data_set_name + '_train.dat'
    path_test = '../data/data/' + data_set_name + '_test.dat'
    dataset_class = load_data(path_train=path_train, path_test=path_test,
                              header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)
    attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                        "../data/data/" + data_set_name + "_target_users"]
    attack_info = load_attack_info(*attack_info_path)
    return dataset_class, attack_info


def baseline_attack(dataset_class, attack_info, attack_method, target_id, bandwagon_selected,
                    fixed_filler_indicator=None):
    """load data"""
    selected_ids, target_users = attack_info[target_id]
    attack_model, attack_num, filler_num = attack_method.split('_')
    attack_num, filler_num = int(attack_num), int(filler_num)

    """attack class"""
    global_mean, global_std, item_means, item_stds = dataset_class.get_all_mean_std()
    baseline_attacker = BaselineAttack(attack_num, filler_num, dataset_class.n_items, target_id,
                                       global_mean, global_std, item_means, item_stds, 5.0, 1.0,
                                       fixed_filler_indicator=fixed_filler_indicator)
    # fake profile array
    fake_profiles = None
    if attack_model == "random":
        fake_profiles = baseline_attacker.RandomAttack()
    elif attack_model == "bandwagon":
        fake_profiles = baseline_attacker.BandwagonAttack(bandwagon_selected)
    elif attack_model == "average":
        fake_profiles = baseline_attacker.AverageAttack()
    elif attack_model == "segment":
        fake_profiles = baseline_attacker.SegmentAttack(selected_ids)
    else:
        print('attack_method error')
        exit()
    return fake_profiles


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='automotive', help='filmTrust/ml100k/grocery')

    parser.add_argument('--attack_methods', type=str, default='average',
                        help='average,segment,random,bandwagon')

    # filmTrust:random = [5, 395, 181, 565, 254]    tail = [601, 623, 619, 64, 558]
    # ml100k:random = [62, 1077, 785, 1419, 1257]   tail = [1319, 1612, 1509, 1545, 1373]
    # 1166,1574,759,494,549,1272,1728,1662,450,1456,595,566,764,1187,1816,1478,1721,2294,2413,1148
    # 62,1077,785,1419,1257,1319,1612,1509,1545,1373
    # 88,22,122,339,1431,1141,1656,477,1089,866
    parser.add_argument('--targets', type=str, default='88,22,122,339,1431,1141,1656,477,1089,866',
                        help='attack_targets')

    parser.add_argument('--attack_num', type=int, default=50, help='fixed 50')

    parser.add_argument('--filler_num', type=int, default=4, help='90 for ml100k,36 for filmTrust')
    parser.add_argument('--bandwagon_selected', type=str, default='180,99,49',
                        help='180,99,49 for ml100k,103,98,115 for filmTrust')
    #
    parser.add_argument('--sample_filler', type=int, default=1, help='sample filler')
    #

    args = parser.parse_args()
    #
    args.attack_methods = args.attack_methods.split(',')
    args.targets = list(map(int, args.targets.split(',')))
    args.bandwagon_selected = list(map(int, args.bandwagon_selected.split(',')))
    return args


if __name__ == '__main__':
    """parse args"""
    args = parse_arg()

    """attack"""
    dataset_class, attack_info = get_data(args.dataset)

    for target_id in args.targets:

        attackSetting_path = '_'.join(map(str, [args.dataset, args.attack_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        if args.sample_filler:
            gan_attacker = Train_GAN_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                              selected_id_list=attack_info[target_id][0],
                                              filler_num=args.filler_num, attack_num=args.attack_num, filler_method=0)
            _, real_profiles, filler_indicator = gan_attacker.execute(is_train=0, model_path='no',
                                                                      final_attack_setting=[args.attack_num,
                                                                                            None, None])

            np.save(attackSetting_path, [real_profiles, filler_indicator])
        else:
            real_profiles, filler_indicator = np.load(attackSetting_path + '.npy')

        # for attack_method in args.attack_methods:
        #
        #     attack_model = '_'.join([attack_method, str(args.attack_num), str(args.filler_num)])
        #     # fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
        #     #                                 args.bandwagon_selected, filler_indicator)
        #     fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
        #                                     args.bandwagon_selected, None)
        #
        #     ori_path = '../data/data/' + args.dataset + '_train.dat'
        #     dst_path = "../data/data_attacked/" + '_'.join([args.dataset, str(target_id), attack_model]) + "_sample.dat"
        #     attacked_file_writer(ori_path, dst_path, fake_profiles, dataset_class.n_users)
