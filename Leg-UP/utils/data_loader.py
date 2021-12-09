# -*- coding: utf-8 -*-
# @Time       : 2020/11/27 15:34
# @Author     : chensi
# @File       : data_loader.py
# @Software   : PyCharm
# @Desciption : None

import random
import numpy as np
import torch

# tf = None
# try:
#     import tensorflow.compat.v1 as tf
#
#     tf.disable_v2_behavior()
# except:
#     import tensorflow as tf

seed = 1234
random.seed(seed)
np.random.seed(seed)
# tf.set_random_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class DataLoader(object):

    def __init__(self, path_train, path_test, header=None, sep='\t', threshold=4, verbose=False):
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.verbose = verbose

        # load file as dataFrame
        # self.train_data, self.test_data, self.n_users, self.n_items = self.load_file_as_dataFrame()
        # dataframe to matrix
        # self.train_matrix, self.train_matrix_implicit = self.dataFrame_to_matrix(self.train_data)
        # self.test_matrix, self.test_matrix_implicit = self.dataFrame_to_matrix(self.test_data)

    def load_file_as_dataFrame(self):
        # load data to pandas dataframe
        if self.verbose:
            print("\nload data from %s ..." % self.path_train, flush=True)

        train_data = pd.read_csv(self.path_train, sep=self.sep, names=self.header, engine='python')
        train_data = train_data.loc[:, ['user_id', 'item_id', 'rating']]

        if self.verbose:
            print("load data from %s ..." % self.path_test, flush=True)
        test_data = pd.read_csv(self.path_test, sep=self.sep, names=self.header, engine='python').loc[:,
                    ['user_id', 'item_id', 'rating']]
        test_data = test_data.loc[:, ['user_id', 'item_id', 'rating']]

        # data statics

        n_users = max(max(test_data.user_id.unique()), max(train_data.user_id.unique())) + 1
        n_items = max(max(test_data.item_id.unique()), max(train_data.item_id.unique())) + 1

        if self.verbose:
            print("Number of users : %d , Number of items : %d. " % (n_users, n_items), flush=True)
            print("Train size : %d , Test size : %d. " % (train_data.shape[0], test_data.shape[0]), flush=True)

        return train_data, test_data, n_users, n_items

    def dataFrame_to_matrix(self, data_frame, n_users, n_items):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            implicit_r = 1 if r >= self.threshold else 0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(n_users, n_items))
        matrix_implicit = csr_matrix((implicit_rating, (row, col)), shape=(n_users, n_items))
        return matrix, matrix_implicit
