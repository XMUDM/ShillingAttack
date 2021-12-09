# -*- coding: utf-8 -*-
# @Time       : 2019/8/23 10:46
# @Author     : chensi
# @File       : baseline_new.py
# @Software   : PyCharm
# @Desciption : None
import numpy as np
import math


class BaselineAttack:

    def __init__(self, attack_num, filler_num, n_items, target_id,
                 global_mean, global_std, item_means, item_stds, r_max, r_min, fixed_filler_indicator=None):
        #
        self.attack_num = attack_num
        self.filler_num = filler_num
        self.n_items = n_items
        self.target_id = target_id
        self.global_mean = global_mean
        self.global_std = global_std
        self.item_means = item_means
        self.item_stds = item_stds
        self.r_max = r_max
        self.r_min = r_min

        self.fixed_filler_indicator = fixed_filler_indicator

    def RandomAttack(self):
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target
        fake_profiles[:, self.target_id] = self.r_max
        # fillers
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:

                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def BandwagonAttack(self, selected_ids):
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target & selected patch
        fake_profiles[:, [self.target_id] + selected_ids] = self.r_max
        # fillers
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:

                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            ratings = np.random.normal(loc=self.global_mean, scale=self.global_std, size=self.filler_num)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def AverageAttack(self):
        filler_candis = list(set(range(self.n_items)) - {self.target_id})
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target
        fake_profiles[:, self.target_id] = self.r_max
        # fillers
        fn_normal = lambda iid: np.random.normal(loc=self.item_means[iid], scale=self.item_stds[iid], size=1)[0]
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:

                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            ratings = map(fn_normal, fillers)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.r_max, r))
        return fake_profiles

    def SegmentAttack(self, selected_ids):
        filler_candis = list(set(range(self.n_items)) - set([self.target_id] + selected_ids))
        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # target & selected patch
        fake_profiles[:, [self.target_id] + selected_ids] = self.r_max
        # fillers
        for i in range(self.attack_num):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candis, size=self.filler_num, replace=False)
            else:

                fillers = np.where(np.array(self.fixed_filler_indicator[i])== 1)[0]
            fake_profiles[i][fillers] = self.r_min
        return fake_profiles
