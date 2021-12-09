# -*- coding: utf-8 -*-
# @Time       : 2019/8/29 21:51
# @Author     : chensi
# @File       : id_plus_1.py
# @Software   : PyCharm
# @Desciption : None


import numpy as np
import pandas as pd
import os

conf_path = '../config/FAP.conf'

# random_target = [62, 1077, 785, 1419, 1257]
# tail_target = [1319, 1612, 1509, 1545, 1373]
# targets = random_target + tail_target
random = [155, 383, 920, 941, 892]
tail = [1480, 844, 1202, 1301, 2035]
targets = random + tail
attack_methods = ["segment", "average", "random", "bandwagon", "gan"]
for iid in targets:
    for attack_method in attack_methods:
        path = "../dataset/GAN/ciao/ciao_" + str(iid) + "_" + attack_method + "_50_15.dat"
        names = ['userID', 'movieID', 'movieRating']
        data_df = pd.read_csv(path, sep='\t', names=names, engine='python')
        data_df.userID += 1
        data_df.movieID += 1
        dst_path = "../dataset/GAN/ciao_1/ciao_" + str(iid) + "_" + attack_method + "_50_15.dat"
        data_df.to_csv(dst_path, index=False, sep='\t', header=False)
