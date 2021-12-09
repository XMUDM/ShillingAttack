#!/usr/bin/env python
"""Implementation of Item based AutoRec and user based AutoRec.
Reference: Sedhain, Suvash, et al. "Autorec: Autoencoders meet collaborative filtering." Proceedings of the 24th International Conference on World Wide Web. ACM, 2015.
"""

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import time
import numpy as np
import scipy, math

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class IAutoRec():
    def __init__(self, sess, dataset_class, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=500,
                 hidden_neuron=500, verbose=False, T=5, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.hidden_neuron = hidden_neuron
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()
        self.verbose = verbose
        self.T = T
        self.display_step = display_step

        self.train_data = self.dataset_class.train_matrix.toarray()
        self.train_data_mask = scipy.sign(self.train_data)

        print("IAutoRec.",end=' ')
        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        # placeholder
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)
        # Variable
        V = tf.Variable(tf.random_normal([self.hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, self.hidden_neuron], stddev=0.01))
        mu = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
                            tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        loss = float('inf')
        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={
                                        self.rating_matrix: self.dataset_class.train_matrix[:, batch_set_idx].toarray(),
                                        self.rating_matrix_mask: scipy.sign(
                                            self.dataset_class.train_matrix[:, batch_set_idx].toarray()),
                                        self.keep_rate_net: 1
                                    })  # 0.95
        return loss

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.reconstruction[u, i]  # self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            # if epoch % self.T == 0:
                # print("epoch:\t", epoch, "\tloss:\t", loss_cur)
            if abs(loss_cur - loss_prev) < math.exp(-5):
                break
            loss_prev = loss_cur
        rmse, mae = self.test(self.dataset_class.test_matrix_dok)
        print("training done\tRMSE : ", rmse, "\tMAE : ", mae)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        return self.reconstruction[user_id, item_id]
        # if not hasattr(self, 'reconstruction_all'):
        #     self.reconstruction_all = self.sess.run(self.layer_2,
        #                                             feed_dict={self.rating_matrix: self.train_data,
        #                                                        self.rating_matrix_mask: self.train_data_mask,
        #                                                        self.keep_rate_net: 1})
        # return self.reconstruction_all[user_id, item_id]


class UAutoRec():
    def __init__(self, sess, dataset_class, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=200,
                 hidden_neuron=500, verbose=False, T=5, display_step=1000, layer=1):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.hidden_neuron = hidden_neuron
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        print("UAutoRec.")
        # 评分矩阵是IAutoRec的转置
        self.train_data = self.dataset_class.train_matrix.toarray().transpose()
        self.train_data_mask = scipy.sign(self.train_data)

        self.layer = layer

        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        # placeholder
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_item, None])
        if self.layer == 1:
            # Variable
            V = tf.Variable(tf.random_normal([self.hidden_neuron, self.num_item], stddev=0.01))
            W = tf.Variable(tf.random_normal([self.num_item, self.hidden_neuron], stddev=0.01))

            mu = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
            layer_1 = tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix))
            self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
            Loss_norm = tf.square(tf.norm(W)) + tf.square(tf.norm(V))
        elif self.layer == 3:
            V_1 = tf.Variable(tf.random_normal([self.hidden_neuron, self.num_item], stddev=0.01))
            V_2 = tf.Variable(tf.random_normal([self.hidden_neuron // 2, self.hidden_neuron], stddev=0.01))
            V_3 = tf.Variable(tf.random_normal([self.hidden_neuron, self.hidden_neuron // 2], stddev=0.01))
            W = tf.Variable(tf.random_normal([self.num_item, self.hidden_neuron], stddev=0.01))
            mu_1 = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            mu_2 = tf.Variable(tf.random_normal([self.hidden_neuron // 2], stddev=0.01))
            mu_3 = tf.Variable(tf.random_normal([self.hidden_neuron], stddev=0.01))
            b = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
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

    def train(self):
        self.num_training = self.num_user
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        for i in range(total_batch):
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx]
                                               })
        return loss

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            if epoch % self.T == 0:
                print("epoch:\t", epoch, "\tloss:\t", loss_cur)
            if abs(loss_cur - loss_prev) < math.exp(-5):
                break
            loss_prev = loss_cur
        rmse, mae = self.test(self.dataset_class.test_matrix_dok)
        print("training done\tRMSE : ", rmse, "\tMAE : ", mae)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask:
                                                                         self.train_data_mask})
        return self.reconstruction[item_id, user_id]
