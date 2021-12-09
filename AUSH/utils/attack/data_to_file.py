# -*- coding: utf-8 -*-
# @Time       : 2019/8/23 21:17
# @Author     : chensi
# @File       : data_to_file.py
# @Software   : PyCharm
# @Desciption : None

import os
import shutil


def attacked_file_writer(clean_path, attacked_path, fake_profiles, n_users_ori):
    data_to_write = ""
    i = 0
    for fake_profile in fake_profiles:
        injected_iid = fake_profile.nonzero()[0]
        injected_rating = fake_profile[injected_iid]
        data_to_write += ('\n'.join(
            map(lambda x: '\t'.join(map(str, [n_users_ori + i] + list(x))), zip(injected_iid, injected_rating))) + '\n')
        i += 1
    if os.path.exists(attacked_path): os.remove(attacked_path)
    shutil.copyfile(clean_path, attacked_path)
    with open(attacked_path, 'a+')as fout:
        fout.write(data_to_write)


def target_prediction_writer(predictions, hit_ratios, dst_path):
    # uid - rating - HR
    data_to_write = []
    for uid in range(len(predictions)):
        data_to_write.append('\t'.join(map(str, [uid, predictions[uid]] + hit_ratios[uid])))
    with open(dst_path, 'w')as fout:
        fout.write('\n'.join(data_to_write))
