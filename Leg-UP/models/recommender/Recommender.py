# -*- coding: utf-8 -*-
# @Time       : 2020/11/27 17:20
# @Author     : chensi
# @File       : Recommender.py
# @Software   : PyCharm
# @Desciption : None

import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# def available_GPU():
#     import subprocess
#     import numpy as np
#     nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
#     total_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
#     total_GPU = total_GPU_str.split('\n')
#     total_GPU = np.array([int(device_i) for device_i in total_GPU])
#     avail_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
#     avail_GPU = avail_GPU_str.split('\n')
#     avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
#     avail_GPU = avail_GPU / total_GPU
#     return np.argmax(avail_GPU)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"


# try:
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(available_GPU())
# except:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import random
import numpy as np
import torch

tf = None
try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils.data_loader import DataLoader
import numpy as np
import pandas as pd
import argparse, scipy, math
import surprise
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import PredefinedKFold


class Recommender(object):

    def __init__(self):
        self.args = self.parse_args()
        # 路径
        self.train_path = self.args.train_path
        self.test_path = self.args.test_path
        self.model_path = self.args.model_path
        self.target_prediction_path_prefix = self.args.target_prediction_path_prefix
        # 攻击
        self.target_id_list = list(map(int, self.args.target_ids.split(',')))
        self.topk_list = list(map(int, self.args.topk.split(',')))
        #
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_id)
        pass

    @staticmethod
    def parse_args():

        parser = argparse.ArgumentParser(description="Run Recommender.")
        parser.add_argument('--data_set', type=str, default='ml100k')  # , required=True)
        # 路径
        parser.add_argument('--train_path', type=str,
                            default='./data/ml100k/ml100k_train.dat')  # , required=True)
        parser.add_argument('--test_path', type=str,
                            default='./data/ml100k/ml100k_test.dat')  # , required=True)
        parser.add_argument('--model_path', type=str,
                            default='./results/model_saved/automotive/automotive_NeuMF_AUSHplus_round_119')  # , required=True)
        parser.add_argument('--target_prediction_path_prefix', type=str,
                            default='./results/performance/mid_results/ml100k_Recommender')  # , required=True)

        # 攻击
        parser.add_argument('--target_ids', type=str, default='0')  # , required=True)
        parser.add_argument('--topk', type=str, default='5,10,20,50')
        #
        parser.add_argument('--cuda_id', type=int, default=0)
        return parser

    def prepare_data(self):
        self.dataset_class = DataLoader(self.train_path, self.test_path)

        self.train_data_df, self.test_data_df, self.n_users, self.n_items = self.dataset_class.load_file_as_dataFrame()
        self.train_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        self.test_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.test_data_df, self.n_users, self.n_items)
        pass

    def build_network(self):
        print('build Recommender model graph.')
        raise NotImplemented

    def train(self):
        print('train.')
        raise NotImplemented

    def test(self):
        print('test.')
        raise NotImplemented

    def execute(self):
        print('generate target item performace on a trained Recommender model.')
        raise NotImplemented

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        raise NotImplemented

    def generate_target_result(self):
        train_data_array = self.train_matrix.toarray()
        for target_id in self.target_id_list:
            # mask掉已评分用户以及未评分用户的已评分商品
            mask = np.zeros_like(train_data_array)
            mask[np.where(train_data_array[:, target_id])[0]] = float('inf')
            # 找到测试数据
            test_uids, test_iids = np.where((train_data_array + mask) == 0)
            # 预测
            test_predRatings = self.predict(test_uids, test_iids)
            # 构建dataframe
            predResults = pd.DataFrame({'user_id': test_uids,
                                        'item_id': test_iids,
                                        'rating': test_predRatings
                                        })
            # 为每个未评分计算预测分和HR
            predResults_target = np.zeros([len(predResults.user_id.unique()), len(self.topk_list) + 2])
            for idx, (user_id, pred_result) in enumerate(predResults.groupby('user_id')):
                pred_value = pred_result[pred_result.item_id == target_id].rating.values[0]
                sorted_recommend_list = pred_result.sort_values('rating', ascending=False).item_id.values
                new_line = [user_id, pred_value] + [1 if target_id in sorted_recommend_list[:k] else 0 for k in
                                                    self.topk_list]
                predResults_target[idx] = new_line
            np.save('%s_%d' % (self.target_prediction_path_prefix, target_id), predResults_target)


class AutoRec(Recommender):
    def __init__(self):
        super(AutoRec, self).__init__()
        self.restore_model = self.args.restore_model
        self.learning_rate = self.args.learning_rate
        self.epochs = self.args.epoch
        self.batch_size = self.args.batch_size
        self.reg_rate = self.args.reg_rate
        self.verbose = self.args.verbose
        self.T = self.args.T
        #
        self.hidden_neuron = self.args.hidden_neuron
        #
        print("AutoRec.", end=' ')

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--reg_rate', type=float, default=0.1)
        parser.add_argument('--epoch', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=500)
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        parser.add_argument('--display_step', type=int, default=1000)
        #
        parser.add_argument('--hidden_neuron', type=int, default=500)
        #
        return parser

    def prepare_data(self):
        super(AutoRec, self).prepare_data()
        self.train_data_array = self.train_matrix.toarray()
        self.train_data_mask_array = scipy.sign(self.train_data_array)

    def build_network(self):
        raise NotImplemented

    def predict(self, user_id, item_id):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def test(self):
        raise NotImplemented

    def execute(self):
        # 数据准备
        self.prepare_data()

        # tensorflow session
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.sess = sess

            self.build_network()


            init = tf.global_variables_initializer()
            sess.run(init)


            if self.restore_model:
                self.restore(self.model_path)
                print("loading done.")


            else:
                loss_prev = float('inf')
                for epoch in range(self.epochs):
                    loss_cur = self.train()
                    if self.verbose and epoch % self.T == 0:
                        print("epoch:\t", epoch, "\tloss:\t", loss_cur)
                    if abs(loss_cur - loss_prev) < math.exp(-5):
                        break
                    loss_prev = loss_cur


                self.save(self.model_path)
                print("training done.")


            rmse, mae = self.test()
            print("RMSE : %.4f,\tMAE : %.4f" % (rmse, mae))

            self.generate_target_result()

            return


class IAutoRec(AutoRec):
    def __init__(self):
        super(IAutoRec, self).__init__()
        print("IAutoRec.", end=' ')

    @staticmethod
    def parse_args():
        parser = AutoRec.parse_args()
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(IAutoRec, self).prepare_data()

    def build_network(self):
        # placeholder
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)

        # Variable
        V = tf.Variable(tf.random_normal([self.hidden_neuron, self.n_users], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.n_users, self.hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.n_users], stddev=0.01))

        # forward
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)

        # backward
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
                            tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def predict(self, user_id, item_id):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data_array,
                                                                     self.rating_matrix_mask: self.train_data_mask_array,
                                                                     self.keep_rate_net: 1})
        return self.reconstruction[user_id, item_id]

    def train(self):
        total_batch = int(self.n_items / self.batch_size)
        idxs = np.random.permutation(self.n_items)  # shuffled ordering
        loss = []
        for i in range(total_batch):
            batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss_ = self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={
                    self.rating_matrix: self.train_matrix[:, batch_set_idx].toarray(),
                    self.rating_matrix_mask: scipy.sign(self.train_matrix[:, batch_set_idx].toarray()),
                    self.keep_rate_net: 1  # 0.95
                })

            loss.append(loss_)
        return np.mean(loss)

    def test(self):
        self.reconstruction = self.sess.run(self.layer_2,
                                            feed_dict={self.rating_matrix: self.train_data_array,
                                                       self.rating_matrix_mask: self.train_data_mask_array,
                                                       self.keep_rate_net: 1})
        test_data = self.test_matrix.toarray()
        test_data_mask = test_data > 0
        test_data_num = np.sum(test_data_mask)
        #
        mae_matrix = np.abs(test_data - self.reconstruction) * test_data_mask
        rmse_matrix = mae_matrix ** 2
        rmse, mae = np.sum(rmse_matrix) / test_data_num, np.sum(mae_matrix) / test_data_num
        return rmse, mae

    def execute(self):
        super(IAutoRec, self).execute()


class UAutoRec(AutoRec):
    def __init__(self):
        super(UAutoRec, self).__init__()
        #
        self.layer = self.args.layer
        #
        print("UAutoRec.", end=' ')

    @staticmethod
    def parse_args():
        parser = AutoRec.parse_args()
        #
        parser.add_argument('--layer', type=int, default=1)
        #
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(UAutoRec, self).prepare_data()

    def build_network(self):
        # placeholder
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.n_items, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.n_items, None])
        if self.layer == 1:
            # Variable
            V = tf.Variable(tf.random_normal([self.hidden_neuron, self.n_items], stddev=0.01))
            W = tf.Variable(tf.random_normal([self.n_items, self.hidden_neuron], stddev=0.01))

            mu = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            b = tf.Variable(tf.random_normal([self.n_items], stddev=0.01))
            layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix))
            self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
            Loss_norm = tf.square(tf.norm(W)) + tf.square(tf.norm(V))
        elif self.layer == 3:
            V_1 = tf.Variable(tf.random_normal([self.hidden_neuron, self.n_items], stddev=0.01))
            V_2 = tf.Variable(tf.random_normal([self.hidden_neuron // 2, self.hidden_neuron], stddev=0.01))
            V_3 = tf.Variable(tf.random_normal([self.hidden_neuron, self.hidden_neuron // 2], stddev=0.01))
            W = tf.Variable(tf.random_normal([self.n_items, self.hidden_neuron], stddev=0.01))
            mu_1 = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            mu_2 = tf.Variable(tf.random_normal([self.hidden_neuron // 2], stddev=0.01))
            mu_3 = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            b = tf.Variable(tf.random_normal([self.n_items], stddev=0.01))
            #
            layer_1 = tf.sigmoid(tf.matmul(V_1, self.rating_matrix) + tf.expand_dims(mu_1, 1))
            layer_2 = tf.sigmoid(tf.matmul(V_2, layer_1) + tf.expand_dims(mu_2, 1))
            layer_3 = tf.sigmoid(tf.matmul(V_3, layer_2) + tf.expand_dims(mu_3, 1))
            self.layer_2 = tf.matmul(W, layer_3) + tf.expand_dims(b, 1)
            Loss_norm = tf.square(tf.norm(W)) + tf.square(tf.norm(V_1)) + tf.square(tf.norm(V_3)) + tf.square(
                tf.norm(V_3))
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2),
                                self.rating_matrix_mask)))) + self.reg_rate + Loss_norm

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def predict(self, user_id, item_id):
        self.reconstruction = self.sess.run(self.layer_2,
                                            feed_dict={self.rating_matrix: self.train_data_array.transpose(),
                                                       self.rating_matrix_mask: self.train_data_mask_array.transpose()})
        return self.reconstruction.transpose()[user_id, item_id]

    def train(self):
        total_batch = int(self.n_users / self.batch_size)
        idxs = np.random.permutation(self.n_users)  # shuffled ordering
        loss = []
        for i in range(total_batch):
            batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss_ = self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={self.rating_matrix: self.train_data_array.transpose()[:, batch_set_idx],
                           self.rating_matrix_mask: self.train_data_mask_array.transpose()[:, batch_set_idx]
                           })

            loss.append(loss_)
        return np.mean(loss)

    def test(self):
        self.reconstruction = self.sess.run(self.layer_2,
                                            feed_dict={self.rating_matrix: self.train_data_array.transpose(),
                                                       self.rating_matrix_mask:
                                                           self.train_data_mask_array.transpose()})
        test_data = self.test_matrix.toarray().transpose()
        test_data_mask = test_data > 0
        test_data_num = np.sum(test_data_mask)
        #
        mae_matrix = np.abs(test_data - self.reconstruction) * test_data_mask
        rmse_matrix = mae_matrix ** 2
        rmse, mae = np.sum(rmse_matrix) / test_data_num, np.sum(mae_matrix) / test_data_num
        return rmse, mae

    def execute(self):
        super(UAutoRec, self).execute()


class NeuMF(Recommender):
    def __init__(self):
        super(NeuMF, self).__init__()
        self.restore_model = self.args.restore_model
        self.learning_rate = self.args.learning_rate
        self.epochs = self.args.epoch
        self.batch_size = self.args.batch_size
        self.reg_rate = self.args.reg_rate
        self.verbose = self.args.verbose
        self.T = self.args.T
        #
        self.num_factor = self.args.num_factor
        self.num_factor_mlp = self.args.num_factor_mlp
        self.hidden_dimension = self.args.hidden_dimension
        #
        print("NeuMF.")

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--learning_rate', type=float, default=0.5)
        parser.add_argument('--reg_rate', type=float, default=0.01)
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=256)
        #
        parser.add_argument('--num_factor', type=int, default=10)
        parser.add_argument('--num_factor_mlp', type=int, default=64)
        parser.add_argument('--hidden_dimension', type=int, default=10)
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        parser.add_argument('--display_step', type=int, default=1000)
        #
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def build_network(self):
        # self.num_neg_sample = num_neg_sample
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        self.P = tf.Variable(tf.random_normal([self.n_users, self.num_factor], stddev=0.01), dtype=tf.float32)
        self.Q = tf.Variable(tf.random_normal([self.n_items, self.num_factor], stddev=0.01), dtype=tf.float32)

        self.mlp_P = tf.Variable(tf.random_normal([self.n_users, self.num_factor_mlp], stddev=0.01), dtype=tf.float32)
        self.mlp_Q = tf.Variable(tf.random_normal([self.n_items, self.num_factor_mlp], stddev=0.01), dtype=tf.float32)

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        mlp_user_latent_factor = tf.nn.embedding_lookup(self.mlp_P, self.user_id)
        mlp_item_latent_factor = tf.nn.embedding_lookup(self.mlp_Q, self.item_id)

        _GMF = tf.multiply(user_latent_factor, item_latent_factor)

        regularizer = tf.keras.regularizers.l2(self.reg_rate)
        layer_1 = tf.layers.dense(
            inputs=tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1),
            units=self.num_factor_mlp * 2,
            kernel_initializer=tf.random_normal_initializer,
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)

        layer_2 = tf.layers.dense(
            inputs=layer_1,
            units=self.hidden_dimension * 8,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        layer_3 = tf.layers.dense(
            inputs=layer_2,
            units=self.hidden_dimension * 4,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        layer_4 = tf.layers.dense(
            inputs=layer_3,
            units=self.hidden_dimension * 2,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        _MLP = tf.layers.dense(
            inputs=layer_4,
            units=self.hidden_dimension,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer,
            kernel_regularizer=regularizer)

        # self.pred_y = tf.nn.sigmoid(tf.reduce_sum(tf.concat([_GMF, _MLP], axis=1), 1))
        self.pred_rating = tf.reduce_sum(tf.concat([_GMF, _MLP], axis=1), 1)

        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + \
                    self.reg_rate * (tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) +
                                     tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))
        #
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        return self

    def prepare_data(self):
        super(NeuMF, self).prepare_data()
        #
        self.train_matrix_coo = self.train_matrix.tocoo()
        #
        self.user = self.train_matrix_coo.row.reshape(-1)
        self.item = self.train_matrix_coo.col.reshape(-1)
        self.rating = self.train_matrix_coo.data

    def train(self):
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])
        # train
        loss = []
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss_ = self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={self.user_id: batch_user,
                           self.item_id: batch_item,
                           self.y: batch_rating})

            loss.append(loss_)
        return np.mean(loss)

    def test(self):
        test_data = self.test_matrix.todok()
        #
        uids = np.array(list(test_data.keys()))[:, 0]
        iids = np.array(list(test_data.keys()))[:, 1]
        ground_truth = np.array(list(test_data.values()))
        #
        pred_rating = self.predict(uids, iids)
        #
        rmse = np.sqrt(np.mean((pred_rating - ground_truth) ** 2))
        mae = np.mean(np.abs(pred_rating - ground_truth))
        return rmse, mae

    def predict(self, user_ids, item_ids):
        if len(user_ids) < self.batch_size:
            return self.sess.run(self.pred_rating,
                                 feed_dict={
                                     self.user_id: user_ids,
                                     self.item_id: item_ids}
                                 )
        # predict by batch
        total_batch = math.ceil(len(user_ids) / self.batch_size)
        user_ids, item_ids = list(user_ids), list(item_ids)
        pred_rating = []
        for i in range(total_batch):
            batch_user = user_ids[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_ids[i * self.batch_size:(i + 1) * self.batch_size]
            # predict
            batch_pred_rating = self.sess.run(self.pred_rating,
                                              feed_dict={
                                                  self.user_id: batch_user,
                                                  self.item_id: batch_item}
                                              )
            pred_rating += list(batch_pred_rating)
        return pred_rating

    def restore_user_embedding(self):
        # 数据准备
        self.prepare_data()
        self.n_users += 50
        # ================

        attackers = ['AUSHplus_Dis_xiaorong', 'AUSHplus', 'SegmentAttacker', 'BandwagonAttacker',
                     'AverageAttacker', 'RandomAttacker',
                     'AUSH', 'RecsysAttacker',
                     'DCGAN', 'WGAN']
        #
        targets = [62]  # [119, 422, 594, 884, 1593]
        with tf.Session() as sess:
            self.sess = sess

            self.build_network()

            sess.run(tf.global_variables_initializer())

            for target in targets:
                for attacker in attackers:
                    self.model_path = './results/model_saved/ml100k/ml100k_NeuMF_%s_%d' % (attacker, target)
                    if not os.path.exists(self.model_path + '.meta'):
                        continue

                    self.restore(self.model_path)
                    print("loading done.")
                    user_embedding, user_embedding_mlp = self.sess.run([self.P, self.mlp_P])
                    save_path = self.model_path + '_user_embed'
                    save_path = save_path.replace('model_saved', 'performance\mid_results')
                    np.save(save_path, user_embedding)
                    np.save(save_path + '_mlp', user_embedding_mlp)
            return

    def execute(self):

        self.prepare_data()
        # ================

        # tensorflow session
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.sess = sess

            self.build_network()


            init = tf.global_variables_initializer()
            sess.run(init)


            if self.restore_model:
                self.restore(self.model_path)
                print("loading done.")


            else:
                loss_prev = float('inf')
                for epoch in range(self.epochs):
                    loss_cur = self.train()
                    if True:  # self.verbose and epoch % self.T == 0:
                        print("epoch:\t", epoch, "\tloss:\t", loss_cur, flush=True)
                    if abs(loss_cur - loss_prev) < math.exp(-5):
                        break
                    loss_prev = loss_cur


                self.save(self.model_path)
                print("training done.")


            rmse, mae = self.test()
            print("RMSE : %.4f,\tMAE : %.4f" % (rmse, mae))

            self.generate_target_result()

            return


class NNMF(Recommender):
    def __init__(self):
        super(NNMF, self).__init__()

        self.restore_model = self.args.restore_model
        self.learning_rate = self.args.learning_rate
        self.epochs = self.args.epoch
        self.batch_size = self.args.batch_size
        self.reg_rate = self.args.reg_rate
        self.verbose = self.args.verbose
        self.T = self.args.T
        #
        self.num_factor_1 = self.args.num_factor_1
        self.num_factor_2 = self.args.num_factor_2
        self.hidden_dimension = self.args.hidden_dimension
        #
        print("NNMF.")

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--reg_rate', type=float, default=0.1)
        parser.add_argument('--epoch', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=500)
        #
        parser.add_argument('--num_factor_1', type=int, default=100)
        parser.add_argument('--num_factor_2', type=int, default=10)
        parser.add_argument('--hidden_dimension', type=int, default=50)
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        parser.add_argument('--display_step', type=int, default=1000)
        #
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(NNMF, self).prepare_data()
        #
        self.train_matrix_coo = self.train_matrix.tocoo()
        #
        self.user = self.train_matrix_coo.row.reshape(-1)
        self.item = self.train_matrix_coo.col.reshape(-1)
        self.rating = self.train_matrix_coo.data

    def build_network(self):
        print("num_factor_1=%d, num_factor_2=%d, hidden_dimension=%d" % (
            self.num_factor_1, self.num_factor_2, self.hidden_dimension))

        # placeholder
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder("float", [None], 'rating')

        # Variable
        P = tf.Variable(tf.random_normal([self.n_users, self.num_factor_1], stddev=0.01))
        Q = tf.Variable(tf.random_normal([self.n_items, self.num_factor_1], stddev=0.01))

        U = tf.Variable(tf.random_normal([self.n_users, self.num_factor_2], stddev=0.01))
        V = tf.Variable(tf.random_normal([self.n_items, self.num_factor_2], stddev=0.01))

        # forward
        input = tf.concat(values=[tf.nn.embedding_lookup(P, self.user_id),
                                  tf.nn.embedding_lookup(Q, self.item_id),
                                  tf.multiply(tf.nn.embedding_lookup(U, self.user_id),
                                              tf.nn.embedding_lookup(V, self.item_id))
                                  ], axis=1)

        # tf1->tf2
        # regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
        regularizer = tf.keras.regularizers.l2(self.reg_rate)
        layer_1 = tf.layers.dense(inputs=input, units=2 * self.num_factor_1 + self.num_factor_2,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=regularizer)
        layer_2 = tf.layers.dense(inputs=layer_1, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_3 = tf.layers.dense(inputs=layer_2, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_4 = tf.layers.dense(inputs=layer_3, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        output = tf.layers.dense(inputs=layer_4, units=1, activation=None,
                                 bias_initializer=tf.random_normal_initializer,
                                 kernel_initializer=tf.random_normal_initializer,
                                 kernel_regularizer=regularizer)
        self.pred_rating = tf.reshape(output, [-1])

        # backward
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
                            tf.norm(U) + tf.norm(V) + tf.norm(P) + tf.norm(Q))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])
        # train
        loss = []
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss_ = self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={self.user_id: batch_user,
                           self.item_id: batch_item,
                           self.y: batch_rating})

            loss.append(loss_)
        return np.mean(loss)

    def test(self):
        test_data = self.test_matrix.todok()
        #
        uids = np.array(list(test_data.keys()))[:, 0]
        iids = np.array(list(test_data.keys()))[:, 1]
        ground_truth = np.array(list(test_data.values()))
        #
        pred_rating = self.predict(uids, iids)
        #
        rmse = np.sqrt(np.mean((pred_rating - ground_truth) ** 2))
        mae = np.mean(np.abs(pred_rating - ground_truth))
        return rmse, mae

    def predict(self, user_ids, item_ids):
        if len(user_ids) < self.batch_size:
            return self.sess.run(self.pred_rating,
                                 feed_dict={
                                     self.user_id: user_ids,
                                     self.item_id: item_ids}
                                 )
        # predict by batch
        total_batch = math.ceil(len(user_ids) / self.batch_size)
        user_ids, item_ids = list(user_ids), list(item_ids)
        pred_rating = []
        for i in range(total_batch):
            batch_user = user_ids[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_ids[i * self.batch_size:(i + 1) * self.batch_size]
            # predict
            batch_pred_rating = self.sess.run(self.pred_rating,
                                              feed_dict={
                                                  self.user_id: batch_user,
                                                  self.item_id: batch_item}
                                              )
            pred_rating += list(batch_pred_rating)
        return pred_rating

    def execute(self):

        self.prepare_data()

        # tensorflow session
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.sess = sess

            self.build_network()

            init = tf.global_variables_initializer()
            sess.run(init)

            if self.restore_model:
                self.restore(self.model_path)
                print("loading done.")

            else:
                loss_prev = float('inf')
                for epoch in range(self.epochs):
                    loss_cur = self.train()
                    if self.verbose and epoch % self.T == 0:
                        print("epoch:\t", epoch, "\tloss:\t", loss_cur)
                    if abs(loss_cur - loss_prev) < math.exp(-5):
                        break
                    loss_prev = loss_cur

                self.save(self.model_path)
                print("training done.")

            rmse, mae = self.test()
            print("RMSE : %.4f,\tMAE : %.4f" % (rmse, mae))

            self.generate_target_result()

            return


class NRR(Recommender):
    def __init__(self):
        super(NRR, self).__init__()
        self.restore_model = self.args.restore_model
        self.learning_rate = self.args.learning_rate
        self.epochs = self.args.epoch
        self.batch_size = self.args.batch_size
        self.reg_rate = self.args.reg_rate
        self.verbose = self.args.verbose
        self.T = self.args.T
        #
        self.num_factor_user = self.args.num_factor_user
        self.num_factor_item = self.args.num_factor_item
        self.d = self.args.d
        self.hidden_dimension = self.args.hidden_dimension
        #
        print("NRR.")

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--reg_rate', type=float, default=0.1)
        parser.add_argument('--epoch', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=256)
        #
        parser.add_argument('--num_factor_user', type=int, default=40)
        parser.add_argument('--num_factor_item', type=int, default=40)
        parser.add_argument('--d', type=int, default=50)
        parser.add_argument('--hidden_dimension', type=int, default=40)
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        parser.add_argument('--display_step', type=int, default=1000)
        #
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def build_network(self):

        # model dependent arguments
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder("float", [None], 'rating')

        U = tf.Variable(tf.random_normal([self.n_users, self.num_factor_user], stddev=0.01))
        V = tf.Variable(tf.random_normal([self.n_items, self.num_factor_item], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.d]))

        user_latent_factor = tf.nn.embedding_lookup(U, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(V, self.item_id)

        W_User = tf.Variable(tf.random_normal([self.num_factor_user, self.d], stddev=0.01))
        W_Item = tf.Variable(tf.random_normal([self.num_factor_item, self.d], stddev=0.01))

        input = tf.matmul(user_latent_factor, W_User) + tf.matmul(item_latent_factor, W_Item) + b

        regularizer = tf.keras.regularizers.l2(self.reg_rate)
        layer_1 = tf.layers.dense(inputs=input, units=self.d, bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=regularizer)
        layer_2 = tf.layers.dense(inputs=layer_1, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_3 = tf.layers.dense(inputs=layer_2, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_4 = tf.layers.dense(inputs=layer_3, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        output = tf.layers.dense(inputs=layer_4, units=1, activation=None,
                                 bias_initializer=tf.random_normal_initializer,
                                 kernel_initializer=tf.random_normal_initializer,
                                 kernel_regularizer=regularizer)
        self.pred_rating = tf.reshape(output, [-1])

        # print(np.shape(output))
        reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
                            tf.norm(U) + tf.norm(V) + tf.norm(b) + tf.norm(W_Item) + tf.norm(W_User))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def prepare_data(self):
        super(NRR, self).prepare_data()
        #
        self.train_matrix_coo = self.train_matrix.tocoo()
        #
        self.user = self.train_matrix_coo.row.reshape(-1)
        self.item = self.train_matrix_coo.col.reshape(-1)
        self.rating = self.train_matrix_coo.data

    def train(self):
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])

        # train
        loss = []
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss_ = self.sess.run([self.optimizer, self.loss],
                                     feed_dict={self.user_id: batch_user,
                                                self.item_id: batch_item,
                                                self.y: batch_rating})

            loss.append(loss_)
        return np.mean(loss)

    def test(self):
        test_data = self.test_matrix.todok()
        #
        uids = np.array(list(test_data.keys()))[:, 0]
        iids = np.array(list(test_data.keys()))[:, 1]
        ground_truth = np.array(list(test_data.values()))
        #
        pred_rating = self.predict(uids, iids)
        #
        rmse = np.sqrt(np.mean((pred_rating - ground_truth) ** 2))
        mae = np.mean(np.abs(pred_rating - ground_truth))
        return rmse, mae

    def predict(self, user_ids, item_ids):
        if len(user_ids) < self.batch_size:
            return self.sess.run(self.pred_rating,
                                 feed_dict={
                                     self.user_id: user_ids,
                                     self.item_id: item_ids}
                                 )
        # predict by batch
        total_batch = math.ceil(len(user_ids) / self.batch_size)
        user_ids, item_ids = list(user_ids), list(item_ids)
        pred_rating = []
        for i in range(total_batch):
            batch_user = user_ids[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_ids[i * self.batch_size:(i + 1) * self.batch_size]
            # predict
            batch_pred_rating = self.sess.run(self.pred_rating,
                                              feed_dict={
                                                  self.user_id: batch_user,
                                                  self.item_id: batch_item}
                                              )
            pred_rating += list(batch_pred_rating)
        return pred_rating

    def execute(self):

        self.prepare_data()

        # tensorflow session
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.sess = sess

            self.build_network()

            init = tf.global_variables_initializer()
            sess.run(init)

            if self.restore_model:
                self.restore(self.model_path)
                print("loading done.")

            else:
                loss_prev = float('inf')
                for epoch in range(self.epochs):
                    loss_cur = self.train()
                    if True:  # self.verbose and epoch % self.T == 0:
                        print("epoch:\t", epoch, "\tloss:\t", loss_cur)
                    if abs(loss_cur - loss_prev) < math.exp(-5):
                        break
                    loss_prev = loss_cur

                self.save(self.model_path)
                print("training done.")

            rmse, mae = self.test()
            print("RMSE : %.4f,\tMAE : %.4f" % (rmse, mae))

            self.generate_target_result()

            return


class RecommenderOnSurprice(Recommender):

    def __init__(self):
        super(RecommenderOnSurprice, self).__init__()

        print("CF build by surprise.")

    def prepare_data(self):
        super(RecommenderOnSurprice, self).prepare_data()

        reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 5))
        data = Dataset.load_from_folds([(self.train_path, self.test_path)], reader=reader)
        trainset, testset = None, None
        pkf = PredefinedKFold()
        for trainset_, testset_ in pkf.split(data):
            trainset, testset = trainset_, testset_
        self.trainset, self.testset = trainset, testset

    def build_network(self):
        print('build_network')
        self.model = None
        raise NotImplemented

    def predict(self, user_ids, item_ids):
        fn_pred = lambda x: self.model.predict(str(x[0]), str(x[1]), r_ui=0).est
        pred_ratings = list(map(fn_pred, zip(user_ids, item_ids)))
        return pred_ratings

    def train(self):
        self.model.fit(self.trainset)
        return

    def test(self):
        preds = self.model.test(self.testset)
        rmse = accuracy.rmse(preds, verbose=True)
        print("RMSE : %.4f" % (rmse))
        return

    def execute(self):

        self.prepare_data()

        self.build_network()

        self.train()

        self.test()

        self.generate_target_result()
        return


class KNN(RecommenderOnSurprice):
    def __init__(self):
        super(KNN, self).__init__()
        self.user_based = self.args.user_based
        self.dis_method = self.args.dis_method
        self.k = self.args.k
        print("KNN.")

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--user_based', type=int, default=0)  # 1
        parser.add_argument('--dis_method', type=str, default='msd')
        parser.add_argument('--k', type=int, default=50)  # 20
        #

        args, _ = parser.parse_known_args()
        return args

    def build_network(self):
        sim_options = {'user_based': self.user_based, 'name': self.dis_method}
        self.model = surprise.KNNBasic(sim_options=sim_options, k=self.k)


class NMF(RecommenderOnSurprice):
    def __init__(self):
        super(NMF, self).__init__()
        self.n_factors = self.args.n_factors
        print("NMF.")

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--n_factors', type=int, default=25)

        # return parser.parse_args()
        args, _ = parser.parse_known_args()
        return args

    def build_network(self):
        self.model = surprise.NMF(n_factors=self.n_factors)


class SVD(RecommenderOnSurprice):
    def __init__(self):
        super(SVD, self).__init__()
        self.n_factors = self.args.n_factors
        print('SVD.')

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        parser.add_argument('--n_factors', type=int, default=25)
        #
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def build_network(self):
        self.model = surprise.SVD(n_factors=self.n_factors)


class SlopeOne(RecommenderOnSurprice):
    def __init__(self):
        super(SlopeOne, self).__init__()
        # self.n_factors = self.args.n_factors
        print('SlopeOne.')

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()
        #
        # parser.add_argument('--n_factors', type=int, default=25)
        #
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def build_network(self):
        self.model = surprise.SlopeOne()


class CoClustering(RecommenderOnSurprice):
    def __init__(self):
        super(CoClustering, self).__init__()
        print('CoClustering.')

    @staticmethod
    def parse_args():
        parser = Recommender.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def build_network(self):
        self.model = surprise.CoClustering()
