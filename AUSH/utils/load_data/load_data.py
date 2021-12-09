# -*- coding: utf-8 -*-
# @Time       : 2019/8/22 10:07
# @Author     : chensi
# @File       : load_data_new.py
# @Software   : PyCharm
# @Desciption : None


import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix


class load_data():

    def __init__(self, path_train, path_test,
                 header=None, sep='\t', threshold=4, print_log=True):
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.print_log = print_log

        self._main_load()

    def _main_load(self):
        # load data
        self._load_file()
        #
        # dataframe to matrix
        self.train_matrix, self.train_matrix_implicit = self._data_to_matrix(self.train_data)
        self.test_matrix, self.test_matrix_implicit = self._data_to_matrix(self.test_data)

    def _load_file(self):
        if self.print_log:
            print("load train/test data\t:\n", self.path_train)
        self.train_data = pd.read_csv(self.path_train, sep=self.sep, names=self.header, engine='python').loc[:,
                          ['user_id', 'item_id', 'rating']]
        self.test_data = pd.read_csv(self.path_test, sep=self.sep, names=self.header, engine='python').loc[:,
                         ['user_id', 'item_id', 'rating']]

        self.n_users = len(set(self.test_data.user_id.unique()) | set(self.train_data.user_id.unique()))
        self.n_items = len(set(self.test_data.item_id.unique()) | set(self.train_data.item_id.unique()))

        if self.print_log:
            print("Number of users:", self.n_users, ",Number of items:", self.n_items, flush=True)
            print("Train size:", self.train_data.shape[0], ",Test size:", self.test_data.shape[0], flush=True)

    def _data_to_matrix(self, data_frame):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            implicit_r = 1 if r >= self.threshold else 0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(self.n_users, self.n_items))
        matrix_implicit = csr_matrix((implicit_rating, (row, col)), shape=(self.n_users, self.n_items))
        return matrix, matrix_implicit

    def get_global_mean_std(self):
        return self.train_matrix.data.mean(), self.train_matrix.data.std()

    def get_all_mean_std(self):
        flag = 1
        for v in ['global_mean', 'global_std', 'item_means', 'item_stds']:
            if not hasattr(self, v):
                flag = 0
                break
        if flag == 0:
            global_mean, global_std = self.get_global_mean_std()
            item_means, item_stds = [global_mean] * self.n_items, [global_std] * self.n_items
            train_matrix_t = self.train_matrix.transpose()
            for iid in range(self.n_items):
                item_vec = train_matrix_t.getrow(iid).toarray()[0]
                ratings = item_vec[np.nonzero(item_vec)]
                if len(ratings) > 0:
                    item_means[iid], item_stds[iid] = ratings.mean(), ratings.std()
            self.global_mean, self.global_std, self.item_means, self.item_stds \
                = global_mean, global_std, item_means, item_stds
        return self.global_mean, self.global_std, self.item_means, self.item_stds

    def get_item_pop(self):
        # item_pops = [0] * self.n_items
        # train_matrix_t = self.train_matrix.transpose()
        # for iid in range(self.n_items):
        #     item_vec = train_matrix_t.getrow(iid).toarray()[0]
        #     item_pops[iid] = len(np.nonzero(item_vec)[0])
        item_pops_dict = dict(self.train_data.groupby('item_id').size())
        item_pops = [0] * self.n_items
        for iid in item_pops_dict.keys():
            item_pops[iid] = item_pops_dict[iid]
        return item_pops

    def get_user_nonrated_items(self):
        non_rated_indicator = self.train_matrix.toarray()
        non_rated_indicator[non_rated_indicator > 0] = 1
        non_rated_indicator = 1 - non_rated_indicator
        user_norated_items = {}
        for uid in range(self.n_users):
            user_norated_items[uid] = list(non_rated_indicator[uid].nonzero()[0])
        return user_norated_items

    def get_item_nonrated_users(self, item_id):
        item_vec = np.squeeze(self.train_matrix[:, item_id].toarray())
        # item_vec = self.train_matrix.toarray().transpose()[item_id]
        item_vec[item_vec > 0] = 1
        non_rated_indicator = 1 - item_vec
        return list(non_rated_indicator.nonzero()[0])
