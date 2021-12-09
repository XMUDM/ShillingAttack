# -*- coding: utf-8 -*-
# @Time       : 2019/8/24 10:45
# @Author     : chensi
# @File       : trainer.py
# @Software   : PyCharm
# @Desciption : None

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import numpy as np
import random, time, copy, os
from model.attack_model.gan_attack.models import GAN_Attacker
from utils.load_data.load_attack_info import *


class Train_GAN_Attacker:
    def __init__(self, dataset_class, params_D, params_G, target_id, selected_id_list,
                 filler_num, attack_num, filler_method):
        # TODO:init refine
        # data set info
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items

        # attack info
        self.target_id = target_id
        self.selected_id_list = selected_id_list
        self.selected_num = len(self.selected_id_list)
        self.filler_num = filler_num
        self.attack_num = attack_num
        self.filler_method = filler_method

        # model params
        self.totalEpochs = 150
        self.ZR_ratio = 0.5
        # G
        if params_G is None:
            # MLP structure
            self.hiddenDim_G = 400
            # optimize params
            self.reg_G = 0.0001
            self.lr_G = 0.01
            self.opt_G = 'adam'
            self.step_G = 1
            self.batchSize_G = 128 * 2
            self.batchNum_G = 10
            self.G_loss_weights = [1, 1, 1, 1]
            self.decay_g = 3
        else:
            self.hiddenDim_G, self.hiddenLayer_G, self.scale, \
            self.reg_G, self.lr_G, self.opt_G, self.step_G, self.batchSize_G, self.batchNum_G, self.G_loss_weights = params_G

        if params_D is None:
            # MLP structure
            self.hiddenDim_D = 150
            self.hiddenLayer_D = 3
            # optimize params
            self.reg_D = 1e-05
            self.lr_D = 0.0001
            self.opt_D = 'adam'
            self.step_D = 1
            self.batchSize_D = 64
        else:
            self.hiddenDim_D, self.hiddenLayer_D, \
            self.reg_D, self.lr_D, self.opt_D, self.step_D, self.batchSize_D = params_D
        #
        self.log_dir = '_'.join(
            list(map(str, self.G_loss_weights + [self.step_G, self.step_D, self.ZR_ratio, str(target_id)])))

    def train_gan(self):
        for epoch in range(self.totalEpochs):
            self.epoch = epoch
            with open(self.log_path, "a+") as fout:
                fout.write("epoch:" + str(epoch) + "\n")
                fout.flush()

            for epoch_D in range(self.step_D):
                self.epoch_D = epoch_D
                loss_D, a, b = self.train_D()
                print('D', epoch_D, ':', round(loss_D, 5), a, end="")
                print(b[0])
                with open(self.log_path, "a+") as fout:
                    log_tmp = 'D' + str(epoch_D) + ':' + str(round(loss_D, 5)) + str(a) + str(b[0])
                    fout.write(log_tmp + "\n")
                    fout.flush()

            for epoch_G in range(self.step_G):
                self.epoch_G = epoch_G
                loss_G, loss_G_array, g_out_seed, log_info = self.train_G()
                with open(self.log_path, "a+") as fout:
                    log_tmp = 'G' + str(epoch_G) + ':' + str(round(loss_G, 5)) \
                              + str(loss_G_array) + str(g_out_seed) + str(log_info)
                    fout.write(log_tmp + "\n")
                    fout.flush()
                print('G', epoch_G, ':', round(loss_G, 5), loss_G_array, g_out_seed, log_info)

    def execute(self, is_train, model_path, final_attack_setting):
        self.log_path = 'logs/' + self.log_dir + '/' + "training_log.log"

        with tf.Graph().as_default():
            self._data_preparation()
            self._build_graph()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # 训练或恢复模型
            if is_train == 0:
                if model_path != 'no':
                    self.restore(model_path)
            else:
                self.wirter = tf.summary.FileWriter('logs/' + self.log_dir + '/', self.sess.graph)
                self.train_gan()
                self.save(model_path)
            # 生成攻击文件
            fake_profiles, real_profiles_, filler_indicator_ \
                = self.fake_profiles_generator(final_attack_setting)
            return fake_profiles, real_profiles_, filler_indicator_

    def fake_profiles_generator(self, final_attack_setting):
        fake_num, real_vector, filler_indicator = final_attack_setting

        # input filler
        if real_vector is None or filler_indicator is None:
            batchList = self.batchList.copy()
            while fake_num > len(batchList):
                batchList += batchList
            random.shuffle(batchList)
            sampled_index = batchList[:fake_num]
            real_vector = self.dataset_class.train_matrix[sampled_index].toarray()
            filler_indicator = self.filler_sampler(sampled_index)

        # output fake profiles
        fake_profiles = self.sess.run(self.fakeData, feed_dict={self.G_input: real_vector,
                                                                self.filler_dims: filler_indicator})
        return fake_profiles, real_vector, filler_indicator
        # if return_real_filler == 0:
        #     return fake_profiles
        # else:
        #     # batchList = self.batchList.copy()
        #     # while fake_num > len(batchList):
        #     #     batchList += batchList
        #     # random.shuffle(batchList)
        #     # sampled_index = batchList[:fake_num]
        #     # # real_profiles = self.train_matrix[sampled_index]
        #     # real_profiles = self.dataset_class.train_matrix[sampled_index].toarray()
        #     # filler_indicator = np.array(self.filler_sampler(sampled_index))
        #     # for idx in range(filler_indicator.shape[0]):
        #     #     filler_indicator[idx][self.selected_id_list + [self.target_id]] = 1
        #     # return fake_profiles, real_profiles * filler_indicator
        #     return fake_profiles, real_vector, filler_indicator

    def _build_graph(self):
        self.filler_dims = tf.placeholder(tf.float32, [None, self.num_item])  # filler = 1, otherwise 0
        self.selected_dims = tf.squeeze(
            tf.reduce_sum(tf.one_hot([self.selected_id_list], self.num_item, dtype=tf.float32), 1))

        self.models = GAN_Attacker()
        # G
        with tf.name_scope("Generator"):
            self.G_input = tf.placeholder(tf.float32, [None, self.num_item], name="G_input")
            self.rating_matrix_mask = tf.placeholder(tf.float32, [None, self.num_item])  # rated = 1, otherwise 0
            self.G_output, self.G_L2norm = self.models.GEN(self.G_input * self.filler_dims, self.num_item,
                                                           self.hiddenDim_G, self.selected_num, 'sigmoid',
                                                           decay=self.decay_g, name="gen")

        with tf.name_scope("Fake_Data"):
            selected_patch = None
            for i in range(self.selected_num):
                one_hot = tf.one_hot(self.selected_id_list[i], self.num_item, dtype=tf.float32)
                mask = tf.boolean_mask(self.G_output, tf.one_hot(i, self.selected_num, dtype=tf.int32), axis=1)
                if i == 0:
                    selected_patch = one_hot * mask
                else:
                    selected_patch += one_hot * mask
            self.fakeData = selected_patch + self.target_patch + self.G_input * self.filler_dims
        # D
        with tf.name_scope("Discriminator"):
            self.realData_ = tf.placeholder(tf.float32, shape=[None, self.num_item], name="real_data")
            self.filler_dims_D = tf.placeholder(tf.float32, [None, self.num_item])  # filler = 1, otherwise 0
            self.realData = self.realData_ * (self.filler_dims_D + self.selected_dims)

            self.D_real = self.models.DIS(self.realData * self.target_mask, self.num_item * 1, self.hiddenDim_D,
                                          'sigmoid', self.hiddenLayer_D)

            self.D_fake = self.models.DIS(self.fakeData * self.target_mask, self.num_item * 1, self.hiddenDim_D,
                                          'sigmoid', self.hiddenLayer_D, _reuse=True)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

        # define loss & optimizer for G
        with tf.name_scope("loss_G"):
            self.g_loss_gan = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
            self.g_loss_seed = tf.reduce_mean(
                tf.reduce_mean(tf.square(self.G_output - 5.0), 1, keepdims=True))
            self.g_loss_reconstruct_seed = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.fakeData - self.G_input) * self.rating_matrix_mask * self.selected_dims,
                              1, keepdims=True))
            self.g_loss_l2 = self.reg_G * self.G_L2norm

            self.g_loss_list = [self.g_loss_gan, self.g_loss_seed,
                                self.g_loss_reconstruct_seed, self.g_loss_l2]
            self.g_loss = sum(self.g_loss_list[i] * self.G_loss_weights[i] for i in range(len(self.G_loss_weights)))

        # tensorboard summary
        self.add_loss_summary(type='G')

        with tf.name_scope("optimizer_G"):
            if self.opt_G == 'sgd':
                self.trainer_G = tf.train.GradientDescentOptimizer(self.lr_G).minimize(self.g_loss,
                                                                                       var_list=self.g_vars,
                                                                                       name="GradientDescent_G")
            elif self.opt_G == 'adam':
                self.trainer_G = tf.train.AdamOptimizer(self.lr_G).minimize(self.g_loss, var_list=self.g_vars,
                                                                            name="Adam_G")

        # define loss & optimizer for D

        with tf.name_scope("loss_D"):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)),
                name="loss_real")
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)),
                name="loss_fake")
            D_L2norm = 0
            for pr in self.d_vars:
                D_L2norm += tf.nn.l2_loss(pr)
            self.d_loss = d_loss_real + d_loss_fake + self.reg_D * D_L2norm
            self.d_loss_real, self.d_loss_fake, self.D_L2norm = d_loss_real, d_loss_fake, D_L2norm
        with tf.name_scope("optimizer_D"):
            if self.opt_D == 'sgd':
                self.trainer_D = tf.train.GradientDescentOptimizer(self.lr_D).minimize(self.d_loss,
                                                                                       var_list=self.d_vars,
                                                                                       name="GradientDescent_D")
            elif self.opt_D == 'adam':
                self.trainer_D = tf.train.AdamOptimizer(self.lr_D).minimize(self.d_loss, var_list=self.d_vars,
                                                                            name="Adam_D")

    def _data_preparation(self):
        self.target_patch = tf.one_hot(self.target_id, self.num_item, dtype=tf.float32) * 5
        self.target_mask = 1 - tf.one_hot(self.target_id, self.num_item, dtype=tf.float32)
        # prepare train data
        # self.train_matrix = self.dataset_class.train_matrix.toarray().astype(np.float32)
        # self.train_mask = self.train_matrix.copy()
        # self.train_mask[self.train_mask > 0] = 1
        self.filler_candi_set = set(range(self.num_item)) - set(self.selected_id_list + [self.target_id])
        self.filler_candi_list = list(self.filler_candi_set)

        # sample filler>filler num
        self.batchList = []
        for i in range(self.num_user):
            set_rated = set(self.dataset_class.train_matrix[i].toarray()[0].nonzero()[0])
            # set_rated = set(self.train_mask[i].nonzero()[0])
            if len(self.filler_candi_set & set_rated) < self.filler_num: continue
            self.batchList.append(i)

        # 没有在train set对target item评分的用户，用来算all user的pred shift
        self.non_rated_users = self.dataset_class.get_item_nonrated_users(self.target_id)
        # item pop/avg
        self.item_pop = np.array(self.dataset_class.get_item_pop())
        _, _, self.item_avg, _ = self.dataset_class.get_all_mean_std()
        self.item_avg = np.array(self.item_avg)

        # big cap
        if self.filler_method == 3:
            print("\n==\n==\n修改路径！！\n==\n")
            attack_info_path = ["../data/data/filmTrust_selected_items", "../data/data/filmTrust_selected_items"]
            attack_info = load_attack_info(*attack_info_path)
            target_users = attack_info[self.target_id][1]
            uid_values = self.dataset_class.train_data.user_id.values
            idxs = [idx for idx in range(len(uid_values)) if uid_values[idx] in target_users]
            iid_values = self.dataset_class.train_data.loc[idxs, 'item_id']
            iid_values = iid_values.tolist()
            from collections import Counter
            iid_values = Counter(iid_values)
            self.item_big_cap = np.array([iid_values.get(iid, 0.5) for iid in range(self.num_item)])

    def train_G(self):
        t1 = time.time()
        random.seed(int(t1))
        random.shuffle(self.batchList)
        #
        batch_real_vector = None
        batch_run_res = None
        #
        total_loss_g = 0
        total_loss_array = np.array([0., 0., 0., 0.])
        total_batch = int(len(self.batchList) / self.batchSize_G) + 1
        for batch_id in range(total_batch):
            if batch_id == total_batch - 1:
                batch_index = self.batchList[batch_id * self.batchSize_G:]
            else:
                batch_index = self.batchList[batch_id * self.batchSize_G: (batch_id + 1) * self.batchSize_G]

            batch_size = len(batch_index)
            # batch_real_vector = self.train_matrix[batch_index]
            batch_real_vector = self.dataset_class.train_matrix[batch_index].toarray()
            # batch_mask = self.train_mask[batch_index]
            batch_mask = batch_real_vector.copy()
            batch_mask[batch_mask > 0] = 1

            # sample zero for zero reconstruction
            batch_mask_ZR = batch_mask.copy()
            if self.ZR_ratio > 0:
                for idx in range(batch_size):
                    batch_mask_ZR[idx][self.selected_id_list] = \
                        [1 if i == 1 or random.random() < self.ZR_ratio else 0 for i in
                         batch_mask_ZR[idx][self.selected_id_list]]

            # sample fillers randomly
            batch_filler_indicator = self.filler_sampler(batch_index)

            batch_run_res = self.sess.run(
                [self.trainer_G, self.g_loss] + self.g_loss_list + [self.G_output, self.G_loss_merged],
                feed_dict={self.G_input: batch_real_vector,
                           self.filler_dims: batch_filler_indicator,
                           self.rating_matrix_mask: batch_mask_ZR})  # Update G

            total_loss_g += batch_run_res[1]
            total_loss_array += np.array(batch_run_res[2:2 + len(total_loss_array)])

        self.wirter.add_summary(batch_run_res[-1], self.step_G * self.epoch + self.epoch_G + 1)
        total_loss_array = [round(i, 2) for i in total_loss_array]
        g_out_seed = [round(i, 2) for i in np.mean(batch_run_res[-2], 0)]
        #
        fn_float_to_str = lambda x: str(round(x, 2))
        r = batch_real_vector.transpose()[self.selected_id_list].transpose()
        g = batch_run_res[-2]
        rmse = list(map(fn_float_to_str, np.sum(np.square(np.abs(r - g)), 0)))
        var_col = list(map(fn_float_to_str, np.var(g, 0)))
        self.add_loss_summary(type="var", info=np.var(g, 0))
        var_row = round(np.mean(np.var(g, 1)), 2)
        # var_col_ori = list(map(fn_float_to_str, np.var(r, 0)))
        # var_row_ori = round(np.mean(np.var(r, 1)), 2)
        log_info = "rmse : " + ','.join(rmse)
        log_info += "\tvar_col : " + ','.join(var_col) + "\tvar_row : " + str(var_row)
        # log_info += "\tvar_col_ori : " + ','.join(var_col_ori) + "\tvar_row_ori : " + str(var_row_ori)
        return total_loss_g, total_loss_array, g_out_seed, log_info  # [g_out_seed, mae, [var_col, var_row]]

    def train_D(self):
        """
        每个epoch各产生self.batchSize_D个realData和fakeData
        """
        t1 = time.time()
        random.seed(int(t1))
        random.shuffle(self.batchList)

        total_loss_d, total_loss_d_real, total_loss_d_fake = 0, 0, 0
        #
        batch_filler_indicator = None

        total_batch = int(len(self.batchList) / self.batchSize_D) + 1
        for batch_id in range(total_batch):
            # prepare data
            if batch_id == total_batch - 1:
                batch_index = self.batchList[batch_id * self.batchSize_D:]
            else:
                batch_index = self.batchList[batch_id * self.batchSize_D: (batch_id + 1) * self.batchSize_D]
            batch_size = len(batch_index)
            batch_real_vector = self.dataset_class.train_matrix[batch_index].toarray()
            # batch_real_vector = self.train_matrix[batch_index]
            batch_filler_indicator = self.filler_sampler(batch_index)

            # optimize
            _, total_loss_d_, total_loss_d_real_, total_loss_d_fake_ \
                = self.sess.run([self.trainer_D, self.d_loss, self.d_loss_real, self.d_loss_fake],
                                feed_dict={self.realData_: batch_real_vector,
                                           self.G_input: batch_real_vector,
                                           self.filler_dims: batch_filler_indicator,
                                           self.filler_dims_D: batch_filler_indicator})  # Update D
            total_loss_d += total_loss_d_
            total_loss_d_real += total_loss_d_real_
            total_loss_d_fake += total_loss_d_fake_
        self.add_loss_summary(type="D", info=[total_loss_d, total_loss_d_real, total_loss_d_fake])
        debug_info = [self.G_output, self.fakeData,
                      tf.squeeze(tf.nn.sigmoid(self.D_real)), tf.squeeze(tf.nn.sigmoid(self.D_fake))]
        info = self.sess.run(debug_info, feed_dict={self.realData_: batch_real_vector,
                                                    self.G_input: batch_real_vector,
                                                    self.filler_dims: batch_filler_indicator,
                                                    self.filler_dims_D: batch_filler_indicator})

        D_real, D_fake = info[2:4]
        fake_data = info[1]
        # lower bound
        lower_bound = []
        for v in fake_data:
            t = v.copy()
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            t[self.selected_id_list] = 5.0
            lower_bound.append(t)
        # upper bound
        upper_bound = []
        i = 0
        for v in fake_data:
            t = v.copy()
            t[self.selected_id_list] = batch_real_vector[i][self.selected_id_list]
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            upper_bound.append(t)
            i += 1
        zero_data = []  # fake_data.copy()
        for v in fake_data:
            t = v.copy()
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            t[self.selected_id_list] = 0.0
            zero_data.append(t)
        random_data = []
        for v in fake_data:
            t = v.copy()
            t[self.selected_id_list] = np.random.choice(list([1., 2., 3., 4., 5.]), size=self.selected_num,
                                                        replace=True)
            t[[self.target_id]] = 0.0  # 对判别器mask掉了target信息
            random_data.append(t)

        D_lower_bound = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                                      feed_dict={self.realData_: lower_bound,
                                                 self.filler_dims_D: batch_filler_indicator})
        D_upper_bound = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                                      feed_dict={self.realData_: upper_bound,
                                                 self.filler_dims_D: batch_filler_indicator})

        D_zero = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                               feed_dict={self.realData_: zero_data, self.filler_dims_D: batch_filler_indicator})
        D_random = self.sess.run(tf.squeeze(tf.nn.sigmoid(self.D_real)),
                                 feed_dict={self.realData_: random_data, self.filler_dims_D: batch_filler_indicator})
        # filler=1通常会更假

        d_info = [round(np.mean(D_real), 2), round(np.mean(D_fake), 2),
                  [round(np.mean(D_lower_bound), 2), round(np.mean(D_upper_bound), 2)],
                  round(np.mean(D_zero), 2), round(np.mean(D_random), 2)]
        # s = ["T:", "G:", "s=5:", "s=0:", "s=random:", "s=5,f=1:"]
        # s = ["real:", "fake:", "seed=5:", "seed=0:", "seed=random:", "seed=5,filler=1:"]
        # d_info = ' '.join([str(d_info[i]) for i in range(len(d_info))])  # s[i]+ str(d_info[i])

        #
        fn_float_to_str = lambda x: str(round(x, 2))
        g_out_seed = list(map(fn_float_to_str, np.mean(info[0], 0)))  # [round(i, 2) for i in np.mean(info[0], 0)]

        #

        g = info[0]
        var_col = list(map(fn_float_to_str, np.var(g, 0)))
        var_row = round(np.mean(np.var(g, 1)), 2)
        log_info = "\tg_out_seed:" + ','.join(g_out_seed), "\tvar_col : " + ','.join(var_col) + "\tvar_row : " + str(
            var_row)

        return total_loss_d, d_info, log_info

    def filler_sampler(self, uid_list):
        if self.filler_method == 0:
            batch_filler_indicator = []
            for uid in uid_list:
                # filler_candi = np.array(
                #     list(set(self.filler_candi_list) & set(self.train_mask[uid].nonzero()[0].tolist())))
                filler_candi = np.array(list(set(self.filler_candi_list)
                                             & set(self.dataset_class.train_matrix[uid].toarray()[0].nonzero()[0])))
                #
                if len(filler_candi) > self.filler_num:
                    filler_candi = np.random.choice(filler_candi, size=self.filler_num, replace=False)
                filler_indicator = [1 if iid in filler_candi else 0 for iid in range(self.num_item)]
                batch_filler_indicator.append(filler_indicator)
            return batch_filler_indicator
        else:
            return self.filler_sampler_method(uid_list)

    def filler_sampler_method(self, uid_list):
        batch_filler_indicator = []
        for uid in uid_list:
            # filler_candi = np.array(
            #     list(set(self.filler_candi_list) & set(self.train_mask[uid].nonzero()[0].tolist())))
            filler_candi = np.array(list(set(self.filler_candi_list)
                                         & set(self.dataset_class.train_matrix[uid].toarray()[0].nonzero()[0])))

            if len(filler_candi) > self.filler_num:
                # sample using a specific method
                # -------------------------
                prob = self.item_avg[filler_candi] if self.filler_method == 1 \
                    else self.item_pop[filler_candi] if self.filler_method == 2 \
                    else self.item_big_cap[filler_candi] if self.filler_method == 3 \
                    else None
                prob = None if prob is None else prob / sum(prob)
                # -------------------------

                filler_candi = np.random.choice(filler_candi, size=self.filler_num, replace=False, p=prob)
            filler_indicator = [1 if iid in filler_candi else 0 for iid in range(self.num_item)]
            batch_filler_indicator.append(filler_indicator)
        return batch_filler_indicator

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def add_loss_summary(self, type="G", info=None):
        # , total_loss_g, total_g_loss_gan, total_g_loss_seed, total_g_loss_reconstruct,total_g_loss_l2):
        if type == "G":
            tf.summary.scalar('Generator/adversarial', self.g_loss_gan)
            tf.summary.scalar('Generator/seed', self.g_loss_seed)
            tf.summary.scalar('Generator/selected_reconstruct', self.g_loss_reconstruct_seed)
            tf.summary.scalar('Generator/l2_normal', self.g_loss_l2)
            tf.summary.scalar('Generator/Sum', self.g_loss)
            self.G_loss_merged = tf.summary.merge_all()

        elif type == 'D':
            total_loss_d, total_loss_d_real, total_loss_d_fake = info
            loss_summary = []
            tag_list = ['Discriminator/Sum', 'Discriminator/real', 'Discriminator/fake']
            simple_value_list = [total_loss_d, total_loss_d_real, total_loss_d_fake]
            for i in range(3):
                loss_summary.append(tf.Summary.Value(tag=tag_list[i], simple_value=simple_value_list[i]))
            self.wirter.add_summary(tf.Summary(value=loss_summary), self.epoch * self.step_D + self.epoch_D + 1)
        elif type == 'var':
            var_summary = []
            for i in range(self.selected_num):
                var_summary.append(tf.Summary.Value(tag='Var/' + str(i), simple_value=info[i]))
            self.wirter.add_summary(tf.Summary(value=var_summary), self.step_G * self.epoch + self.epoch_G + 1)
        else:
            print("summary type error")
