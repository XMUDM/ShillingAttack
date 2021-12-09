# -*- coding: utf-8 -*-
# @Time       : 2019/8/25 19:38
# @Author     : chensi
# @File       : main_eval_similarity.py
# @Software   : PyCharm
# @Desciption : None

import numpy as np
from numpy.linalg import *
import scipy.stats
import sys, os, argparse
import pandas as pd

sys.path.append("../")
from test_main.main_baseline_attack import baseline_attack
from test_main.main_gan_attack import gan_attack
from test_main.main_gan_attack_baseline import gan_attack as gan_attack_baseline
from utils.load_data.load_data import *
from utils.load_data.load_attack_info import load_attack_info
from model.attack_model.gan_attack.trainer import Train_GAN_Attacker


def eval_eigen_value(profiles):
    U_T_U = np.dot(profiles.transpose(), profiles)
    eig_val, _ = eig(U_T_U)
    top_10 = [i.real for i in eig_val[:10]]
    return top_10


def get_item_distribution(profiles):
    # [min(max(0, round(i)), 5) for i in a]
    profiles_T = profiles.transpose()
    fn_count = lambda item_vec: np.array(
        [sum([1 if (min(max(0, round(j)), 5) == i) else 0 for j in item_vec]) for i in range(6)])
    fn_norm = lambda item_vec: item_vec / sum(item_vec)
    item_distribution = np.array(list(map(fn_count, profiles_T)))
    item_distribution = np.array(list(map(fn_norm, item_distribution)))
    return item_distribution


def eval_TVD_JS(P, Q):
    # TVD
    dis_TVD = np.mean(np.sum(np.abs(P - Q) / 2, 1))
    # JS
    fn_KL = lambda p, q: scipy.stats.entropy(p, q)
    M = (P + Q) / 2
    js_vec = []
    for iid in range(P.shape[0]):
        p, q, m = P[iid], Q[iid], M[iid]
        js_vec.append((fn_KL(p, m) + fn_KL(q, m)) / 2)
    dis_JS = np.mean(np.array(js_vec))
    return dis_TVD, dis_JS


def print_eigen_result(real_profiles, fake_profiles_gan, baseline_fake_profiles, baseline_methods):
    top_10_res = []
    top_10_real = eval_eigen_value(real_profiles)
    top_10_res.append("real\t" + '\t'.join(map(str, top_10_real)))
    top_10_baseline = []
    for idx in range(len(baseline_fake_profiles)):
        top_10_baseline.append(eval_eigen_value(baseline_fake_profiles[idx]))
        top_10_res.append(baseline_methods[idx] + "\t" + '\t'.join(map(str, top_10_baseline[-1])))
    top_10_gan = eval_eigen_value(fake_profiles_gan)
    # top_10_sample_5 = eval_eigen_value(fake_profiles_sample_5)
    # top_10_real_sample = eval_eigen_value(real_profiles_gan)
    top_10_res.append("gan\t" + '\t'.join(map(str, top_10_gan)))
    # top_10_res.append("sample_5\t" + '\t'.join(map(str, top_10_sample_5)))
    # top_10_res.append("real_sample\t" + '\t'.join(map(str, top_10_real_sample)))
    print("\n".join(top_10_res))


def get_distance_result(target_id, real_profiles, fake_profiles_gan, baseline_fake_profiles, baseline_methods):
    k = ['target_id', 'attack_method', 'dis_TVD', 'dis_JS']
    v = [[], [], [], []]
    res_dis = []
    real_item_distribution = get_item_distribution(real_profiles)
    # real_gan_item_distribution = get_item_distribution(real_profiles_gan)
    fake_gan_distribution = get_item_distribution(fake_profiles_gan)
    # fake_sample_5_distribution = get_item_distribution(fake_profiles_sample_5)
    # dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, real_gan_item_distribution)
    # res_dis.append('\t'.join(map(str, ["real", "real_gan", dis_TVD, dis_JS])))
    # dis_TVD, dis_JS = eval_TVD_JS(real_gan_item_distribution, fake_gan_distribution)
    # res_dis.append('\t'.join(map(str, ["real_gan", "gan", dis_TVD, dis_JS])))
    # dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, fake_sample_5_distribution)
    # res_dis.append('\t'.join(map(str, ["real", "sample_5", dis_TVD, dis_JS])))
    # dis_TVD, dis_JS = eval_TVD_JS(real_gan_item_distribution, fake_sample_5_distribution)
    # res_dis.append('\t'.join(map(str, ["real_gan", "sample_5", dis_TVD, dis_JS])))
    dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, fake_gan_distribution)
    v[1] += ['gan']
    v[2] += [dis_TVD]
    v[3] += [dis_JS]
    # res_dis.append('\t'.join(map(str, [target_id, "gan", dis_TVD, dis_JS])))
    for idx in range(len(baseline_fake_profiles)):
        dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, get_item_distribution(baseline_fake_profiles[idx]))
        v[1] += [baseline_methods[idx]]
        v[2] += [dis_TVD]
        v[3] += [dis_JS]
        # res_dis.append('\t'.join(map(str, [target_id, baseline_methods[idx], dis_TVD, dis_JS])))
    v[0] = [target_id] * len(v[1])
    result = pd.DataFrame(dict(zip(k, v)))
    # print('\n'.join(res_dis))
    return result


def profiles_generator(target_id, dataset_class, attack_info, bandwagon_selected, sample_num, args, real_profiles,
                       filler_indicator, pre_fix, has_G=False):
    # baseline fake profiles
    baseline_methods = ["segment", "average", "random", "bandwagon"]
    baseline_fake_profiles = []
    for attack_method in baseline_methods:
        attack_model = '_'.join([attack_method, str(sample_num), str(args.filler_num)])
        fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
                                        bandwagon_selected, filler_indicator)
        baseline_fake_profiles.append(fake_profiles)

    for attack_method in baseline_methods:
        attack_model = '_'.join([attack_method, str(sample_num), str(args.filler_num)])
        fake_profiles = baseline_attack(dataset_class, attack_info, attack_model, target_id,
                                        bandwagon_selected, None)
        baseline_fake_profiles.append(fake_profiles)
    baseline_methods = baseline_methods + [i + '_rand' for i in baseline_methods]

    final_attack_setting = [sample_num, real_profiles, filler_indicator]
    # new_baseline
    if has_G:
        for attack_method in ['G0' + pre_fix, 'G1' + pre_fix]:
            baseline_methods.append(attack_method)
            fake_profiles_G, _, _ = gan_attack_baseline(args.dataset, attack_method, target_id, False, 0,
                                                        final_attack_setting=final_attack_setting)
            baseline_fake_profiles.append(fake_profiles_G)

    # gan profiles
    attack_method = "gan" + pre_fix
    fake_profiles_gan, _, _ = gan_attack(args.dataset, attack_method, target_id, False, write_to_file=0,
                                         final_attack_setting=final_attack_setting)
    return fake_profiles_gan, baseline_fake_profiles, baseline_methods


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ml100k',
                        help='input data_set_name,filmTrust or ml100k grocery')

    parser.add_argument('--attack_num', type=int, default=50,
                        help='num of attack fake user,50 for ml100k and filmTrust')

    parser.add_argument('--filler_num', type=int, default=90,
                        help='num of filler items each fake user,90 for ml100k,36 for filmTrust')
    # filmTrust:5,395,181,565,254,601,623,619,64,558 - random*5+tail*5
    # ml100k:62,1077,785,1419,1257,1319,1612,1509,1545,1373 - random*5+tail*5
    parser.add_argument('--targets', type=str, default='62,1077,785,1419,1257,1319,1612,1509,1545,1373', help='attack_targets')
    parser.add_argument('--bandwagon_selected', type=str, default='180,99,49',
                        help='180,99,49 for ml100k,103,98,115 for filmTrust')
    #
    args = parser.parse_args()
    #
    args.targets = list(map(int, args.targets.split(',')))
    args.bandwagon_selected = list(map(int, args.bandwagon_selected.split(',')))
    return args


if __name__ == '__main__':
    """
    step1 - load data
    step2 - 
    step3 - 
    """

    #
    """parse args"""
    args = parse_arg()
    pre_fix = '_' + str(args.attack_num) + '_' + str(args.filler_num)

    """step1 - load data"""
    path_train = "../data/data/" + args.dataset + "_train.dat"
    path_test = "../data/data/" + args.dataset + "_test.dat"
    attack_info_path = ["../data/data/" + args.dataset + "_selected_items",
                        "../data/data/" + args.dataset + "_target_users"]
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)
    attack_info = load_attack_info(*attack_info_path)

    sample_num = dataset_class.n_users
    result = None
    for target_id in args.targets:
        selected = attack_info[target_id][0]

        attackSetting_path = '_'.join(map(str, [args.dataset, sample_num, args.filler_num, target_id]))
        attackSetting_path = "../data/data_attacked/" + attackSetting_path + '_attackSetting'
        gan_attacker = Train_GAN_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                          selected_id_list=selected, filler_num=args.filler_num,
                                          attack_num=args.attack_num, filler_method=0)
        _, real_profiles, filler_indicator = gan_attacker.execute(is_train=0, model_path='no',
                                                                  final_attack_setting=[sample_num, None, None])
        np.save(attackSetting_path, [real_profiles, filler_indicator])

        fake_profiles_gan, baseline_fake_profiles, baseline_methods \
            = profiles_generator(target_id, dataset_class, attack_info, args.bandwagon_selected, sample_num, args,
                                 real_profiles, filler_indicator, pre_fix, has_G=True)


        # result_ = get_distance_result(target_id, real_profiles, fake_profiles_gan, baseline_fake_profiles,
        #                               baseline_methods)
        result_ = get_distance_result(target_id, dataset_class.train_matrix.toarray(), fake_profiles_gan,
                                      baseline_fake_profiles,
                                      baseline_methods)

        result = result_ if result is None else pd.concat([result, result_])
    print(result)
    result.to_excel(args.dataset + '_distance_lianyun.xls', index=False)
