#!/usr/bin/env python
"""Implementation of Neural Network Matrix Factorization.
Reference: Dziugaite, Gintare Karolina, and Daniel M. Roy. "Neural network matrix factorization." arXiv preprint arXiv:1511.06443 (2015).
"""

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import time
import numpy as np
import math

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class NNMF():
    def __init__(self, sess, dataset_class, num_factor_1=100, num_factor_2=10, hidden_dimension=50,
                 learning_rate=0.001, reg_rate=0.01, epoch=500, batch_size=256,
                 show_time=False, T=5, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.dataset_class.test_matrix_dok = self.dataset_class.test_matrix.todok()

        self.num_factor_1 = num_factor_1
        self.num_factor_2 = num_factor_2
        self.hidden_dimension = hidden_dimension
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("NNMF.")

        self.dataset_class_train_matrix_coo = self.dataset_class.train_matrix.tocoo()
        self.user = self.dataset_class_train_matrix_coo.row.reshape(-1)
        self.item = self.dataset_class_train_matrix_coo.col.reshape(-1)
        self.rating = self.dataset_class_train_matrix_coo.data

        self._build_network()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self):
        print("num_factor_1=%d, num_factor_2=%d, hidden_dimension=%d" % (
            self.num_factor_1, self.num_factor_2, self.hidden_dimension))

        # model dependent arguments
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder("float", [None], 'rating')
        # latent feature vectors
        P = tf.Variable(tf.random_normal([self.num_user, self.num_factor_1], stddev=0.01))
        Q = tf.Variable(tf.random_normal([self.num_item, self.num_factor_1], stddev=0.01))
        # latent feature matrix(K=1?)
        U = tf.Variable(tf.random_normal([self.num_user, self.num_factor_2], stddev=0.01))
        V = tf.Variable(tf.random_normal([self.num_item, self.num_factor_2], stddev=0.01))

        input = tf.concat(values=[tf.nn.embedding_lookup(P, self.user_id),
                                  tf.nn.embedding_lookup(Q, self.item_id),
                                  tf.multiply(tf.nn.embedding_lookup(U, self.user_id),
                                              tf.nn.embedding_lookup(V, self.item_id))
                                  ], axis=1)
        #
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
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.user_id: batch_user,
                                                                            self.item_id: batch_item,
                                                                            self.y: batch_rating
                                                                            })
        return loss

    def test(self, test_data):
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict([u], [i])[0]
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
        if type(item_id) != list:
            item_id = [item_id]
        if type(user_id) != list:
            user_id = [user_id] * len(item_id)
        return self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]
