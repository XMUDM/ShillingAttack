# -*- coding: utf-8 -*-
# @Time       : 2020/12/6 12:51
# @Author     : chensi
# @File       : Base_Attacker.py
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


# os.environ["CUDA_VISIBLE_DEVICES"] = str(available_GPU)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import random
import numpy as np
import torch
import time

import os

seed = 1234
tf = None
try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    tf.set_random_seed(seed)
except:
    import tensorflow as tf

    tf.random.set_seed(seed)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils.data_loader import DataLoader
import argparse, os, shutil

PythonCommand = 'D:\Anaconda3\envs\py38_tf2\python'
PythonCommand = PythonCommand if os.path.exists(PythonCommand + '.exe') else 'python'


class Attacker(object):
    def __init__(self):
        self.args = self.parse_args()
        self.data_set = self.args.data_set
        self.target_id = self.args.target_id
        self.attack_num = self.args.attack_num

        self.filler_num = self.args.filler_num
        # self.injected_path = self.args.injected_path


    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Run Attacker.")
        # filmTrust/filmTrust/automotive
        parser.add_argument('--data_set', type=str, default='ml100k')  # , required=True)
        parser.add_argument('--target_id', type=int, default=62)  # , required=True)
        parser.add_argument('--attack_num', type=int, default=50)  # )
        # ml100k:90/automotive:4
        parser.add_argument('--filler_num', type=int, default=36)  # , required=True)
        parser.add_argument('--cuda_id', type=int, default=0)  # , required=True)
        #
        # parser.add_argument('--injected_path', type=str,
        #                     default='./results/data_attacked/ml100k/ml100k_attack_62.data')  # , required=True)
        return parser

    def prepare_data(self):
        self.path_train = './data/%s/%s_train.dat' % (self.data_set, self.data_set)
        path_test = './data/%s/%s_test.dat' % (self.data_set, self.data_set)
        self.dataset_class = DataLoader(self.path_train, path_test)
        self.train_data_df, _, self.n_users, self.n_items = self.dataset_class.load_file_as_dataFrame()

    def build_network(self):
        raise NotImplemented

    def train(self):
        raise NotImplemented

    def test(self, victim='SVD', detect=False, fake_array=None):
        """

        :param victim:
        :param evalutor:
        :return:
        """

        self.generate_injectedFile(fake_array)
        """detect"""
        res_detect_list = self.detect(detect)
        res_detect = '\t'.join(res_detect_list)

        """attack"""
        all_victim_models = ['SVD', 'NMF', 'SlopeOne', 'NeuMF', 'IAutoRec', 'UAutoRec']
        if victim is False:
            res_attack = ''
        elif victim in all_victim_models:
            self.attack(victim)
            res_attack_list = self.evaluate(victim)
            res_attack = '\t'.join(res_attack_list)

        else:
            if victim == 'all':
                victim_models = all_victim_models
            else:
                victim_models = victim.split(',')
            res_attack_list = []
            # SlopeOne,SVD,NMF,IAutoRec,UAutoRec,NeuMF
            for victim_model in victim_models:

                self.attack(victim_model)

                cur_res_list = self.evaluate(victim_model)

                res_attack_list.append('\t:\t'.join([victim_model, '\t'.join(cur_res_list)]))
            res_attack = '\n' + '\n'.join(res_attack_list)
        res = '\t'.join([res_attack, res_detect])
        return res

    def evaluate(self, victim):
        attacker, recommender = self.__class__.__name__, victim
        # #
        args_dict = {
            'data_set': self.data_set,
            'test_path': './data/%s/%s_test.dat' % (self.data_set, self.data_set),
            #
            'target_ids': self.target_id,
            'recommender': recommender,
            'attacker': attacker,
            #
        }
        #
        path_res_before_attack = './results/performance/mid_results/%s/%s_%s_%d.npy' % (
            self.data_set, self.data_set, recommender, self.target_id)

        if not os.path.exists(path_res_before_attack):
            print("path not exists", path_res_before_attack)
            cur_args_dict = {
                'exe_model_lib': 'recommender',
                'exe_model_class': recommender,
                'train_path': './data/%s/%s_train.dat' % (self.data_set, self.data_set),
                'model_path': './results/model_saved/%s/%s_%s' % (self.data_set, self.data_set, recommender),
                'target_prediction_path_prefix': './results/performance/mid_results/%s/%s_%s' % (
                    self.data_set, self.data_set, recommender),
            }
            cur_args_dict.update(args_dict)
            args_str = ' '.join(
                ["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
            return_file = os.popen('%s ./execute_model.py %s' % (PythonCommand, args_str))
            return_str = return_file.read()
        result_list = []

        cur_args_dict = {
            'exe_model_lib': 'evaluator',
            'exe_model_class': 'Attack_Effect_Evaluator',
            'data_path_clean': './results/performance/mid_results/%s/%s_%s_%d.npy' % (
                self.data_set, self.data_set, recommender, self.target_id),
            'data_path_attacked': './results/performance/mid_results/%s/%s_%s_%s_%d.npy' % (
                self.data_set, self.data_set, recommender, attacker, self.target_id),
        }
        cur_args_dict.update(args_dict)
        args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])

        return_file = os.popen('%s ./execute_model.py %s' % (PythonCommand, args_str))
        # time.sleep(5)
        return_str = return_file.read()

        return_str = return_str[return_str.find('result begin') + 13:return_str.find('result end') - 2]
        result_list += [return_str]
        return_file.close()
        # print("========evaluat %s attack %s done.========" % (attacker, recommender))

        return result_list

    def detect(self, detect):
        if not detect:
            return []
        attacker = self.__class__.__name__
        result_list = []

        cur_args_dict = {
            'exe_model_lib': 'evaluator',
            'exe_model_class': 'Attack_Effect_Evaluator',
            'data_path_clean': './data/%s/%s_train.dat' % (self.data_set, self.data_set),
            'data_path_attacked': './results/data_attacked/%s/%s_%s_%d.data' % (
                self.data_set, self.data_set, attacker, self.target_id),
        }
        #
        evalutors = ['Profile_Distance_Evaluator', 'FAP_Detector']
        for evalutor in evalutors:
            cur_args_dict.update({'exe_model_class': evalutor, })
            args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
            return_file = os.popen('%s ./execute_model.py %s' % (PythonCommand, args_str))
            return_str = return_file.read()
            return_str = return_str[return_str.find('result begin') + 13:return_str.find('result end') - 2]
            result_list += [return_str]

        return result_list

    def attack(self, victim):
        attacker, recommender = self.__class__.__name__, victim
        args_dict = {
            'exe_model_lib': 'recommender',
            'exe_model_class': recommender,
            #
            'data_set': self.data_set,
            'train_path': './results/data_attacked/%s/%s_%s_%d.data' \
                          % (self.data_set, self.data_set, self.__class__.__name__, self.target_id),
            'test_path': './data/%s/%s_test.dat' % (self.data_set, self.data_set),
            #
            'target_ids': self.target_id,
            'recommender': recommender,
            # 'attacker': attacker,
            #
            'model_path': './results/model_saved/%s/%s_%s_%s_%d' % (
                self.data_set, self.data_set, recommender, attacker, self.target_id),
            'target_prediction_path_prefix': './results/performance/mid_results/%s/%s_%s_%s' % (
                self.data_set, self.data_set, recommender, attacker),
        }

        args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in args_dict.items()])
        target_file = "%s_%d.npy" % (args_dict['target_prediction_path_prefix'], self.target_id)
        if os.path.exists(target_file):
            os.remove(target_file)

        return_file = os.popen('%s ./execute_model.py %s' % (PythonCommand, args_str))  # popen
        # time.sleep(60 * 3)
        return_str = return_file.read()
        return

    def execute(self):
        raise NotImplemented

    def save(self, path):
        raise NotImplemented

    def restore(self, path):
        raise NotImplemented

    def generate_fakeMatrix(self):
        raise NotImplemented

    def generate_injectedFile(self, fake_array=None):
        if fake_array is None:
            fake_array = self.generate_fakeMatrix()

        # injected_path = './results/data_attacked/ml100k/ml100k_attack_62.data'
        injected_path = './results/data_attacked/%s/%s_%s_%d.data' \
                        % (self.data_set, self.data_set, self.__class__.__name__, self.target_id)

        if os.path.exists(injected_path):
            # print('clear data in %s' % self.injected_path)
            os.remove(injected_path)
        shutil.copyfile(self.path_train, injected_path)

        #
        uids = np.where(fake_array > 0)[0] + self.n_users
        iids = np.where(fake_array > 0)[1]
        values = fake_array[fake_array > 0]
        #
        data_to_write = np.concatenate([np.expand_dims(x, 1) for x in [uids, iids, values]], 1)
        F_tuple_encode = lambda x: '\t'.join(map(str, [int(x[0]), int(x[1]), x[2]]))
        data_to_write = '\n'.join([F_tuple_encode(tuple_i) for tuple_i in data_to_write])
        with open(injected_path, 'a+')as fout:
            fout.write(data_to_write)

        # print('Inject %s successfully' % self.injected_path)
        return

    def visualize(self, results):
        import matplotlib.pyplot as plt
        fig, ax_list = plt.subplots(1, len(results), figsize=(4 * len(results), 4))

        key = sorted(list(results.keys()))
        for idx, ax in enumerate(ax_list):
            if len(results[key[idx]]) == 0:
                continue
            ax.plot(results[key[idx]])
            ax.set_xlabel("iteration")
            ax.set_title(key[idx])
        # plt.show()
        fig_path = "./results/performance/figs/%s/%s_%d.png" \
                   % (self.data_set, self.__class__.__name__, self.target_id)
        plt.savefig(fig_path)


class WGANAttacker(Attacker):
    def __init__(self):
        super(WGANAttacker, self).__init__()
        self.epoch = self.args.epoch
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate
        self.beta1 = self.args.beta1

        self.height = self.args.height
        self.width = self.args.width

        if self.args.data_set == 'ml100k':
            self.height = 29
            self.width = 58
        elif self.args.data_set == 'filmTrust':
            self.height = 29
            self.width = 25
        elif self.args.data_set == 'automotive':
            self.height = 41
            self.width = 45
        elif self.args.data_set == 'ToolHome':
            self.height = 131
            self.width = 78
        elif self.args.data_set == 'GroceryFood':
            self.height = 105
            self.width = 83
        elif self.args.data_set == 'AppAndroid':
            self.height = 119
            self.width = 111
        elif self.args.data_set == 'yelp':
            self.height = 104
            self.width = 111
        else:
            print("Unsupported dataset:", self.args.data_set)
            exit(-1)

        self.z_dim = self.args.z_dim
        self.gf_dim = self.args.gf_dim
        self.df_dim = self.args.df_dim
        self.gfc_dim = self.args.gfc_dim
        self.dfc_dim = self.args.dfc_dim
        self.max_to_keep = self.args.max_to_keep
        #

        self.T = self.args.T
        self.K = self.args.K
        self.alpha = self.args.alpha
        self.eta = self.args.eta

        # batch normalization : deals with poor initialization helps gradient flow
        class batch_norm(object):
            def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
                with tf.variable_scope(name):
                    self.epsilon = epsilon
                    self.momentum = momentum
                    self.name = name

            def __call__(self, x, train=True):
                # return tf.contrib.layers.batch_norm(x,
                #                                     decay=self.momentum,
                #                                     updates_collections=None,
                #                                     epsilon=self.epsilon,
                #                                     scale=True,
                #                                     is_training=train,
                #                                     scope=self.name)
                return tf.layers.batch_normalization(x, training=train)

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        #
        parser.add_argument('--epoch', type=int, default=20)  # 64
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--learning_rate', type=float, default=0.0002)
        parser.add_argument('--beta1', type=float, default=0.5)

        # default is ml100k. will be overwritten in the init function.
        parser.add_argument('--height', type=int, default=29)
        parser.add_argument('--width', type=int, default=58)

        parser.add_argument('--z_dim', type=int, default=100)
        parser.add_argument('--gf_dim', type=int, default=64)
        parser.add_argument('--df_dim', type=int, default=64)
        parser.add_argument('--gfc_dim', type=int, default=1024)
        parser.add_argument('--dfc_dim', type=int, default=1024)
        parser.add_argument('--max_to_keep', type=int, default=1)
        #

        parser.add_argument('--T', type=int, default=10)
        parser.add_argument('--K', type=int, default=5)
        parser.add_argument('--alpha', type=float, default=50.0)
        parser.add_argument('--eta', type=float, default=100.0)

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(WGANAttacker, self).prepare_data()
        train_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        self.train_data_array = train_matrix.toarray()
        self.fake_array = None

    def build_network(self):
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size, self.height, self.width, 1],
                                     name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)

        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        self.d_loss = tf.reduce_mean(tf.square(self.D_logits - self.D_logits_))
        self.g_loss = tf.reduce_mean(tf.square(self.D_logits_))
        self.d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        self.g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

        self.d_optim = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

    def train(self):
        train_idxs = list(range(self.train_data_array.shape[0]))

        for epoch in range(self.epoch):
            np.random.shuffle(train_idxs)
            for i in range(len(train_idxs) // self.batch_size):
                cur_idxs = train_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                batch_inputs = self.train_data_array[cur_idxs]
                # transform range&shape
                batch_inputs = (batch_inputs - 2.5) / 2.5
                pad_length = self.width * self.height - self.n_items
                if pad_length > 0:
                    batch_inputs = np.pad(batch_inputs, ((0, 0), (0, pad_length)), 'constant')
                batch_inputs = np.reshape(batch_inputs, [self.batch_size, self.height, self.width, 1])
                # batch_inputs = np.random.random_sample([self.batch_size, self.height, self.width, 1])
                batch_z = self.gen_random(size=[self.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _ = self.sess.run(self.d_optim, feed_dict={self.inputs: batch_inputs, self.z: batch_z})

                # Update G network
                _ = self.sess.run(self.g_optim, feed_dict={self.z: batch_z})

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                errD = self.d_loss.eval({self.inputs: batch_inputs, self.z: batch_z})
                # errD_real = self.d_loss_real.eval({self.inputs: batch_inputs})
                errG = self.g_loss.eval({self.z: batch_z})

            print("Epoch:[%2d/%2d]d_loss: %.4f, g_loss: %.4f" % (epoch + 1, self.epoch, errD, errG), flush=True)
        batch_z = self.gen_random(size=[self.batch_size, self.z_dim]).astype(np.float32)
        fake_profiles = self.sess.run(self.G, feed_dict={self.z: batch_z})
        fake_profiles = fake_profiles.reshape([self.batch_size, -1])[:self.attack_num, :self.n_items]
        fake_profiles = np.round((fake_profiles * 2.5) + 2.5)

        self.fake_array = self.opt_adv_intent(fake_profiles)
        return self.fake_array

    def opt_adv_intent(self, fake_profiles):
        # filler_num
        filler_indicators = []
        for i in range(self.attack_num):
            fillers_ = np.random.choice(list(range(self.n_items)), self.filler_num, replace=False)
            filler_indicator_ = [1 if iid in fillers_ else 0 for iid in range(self.n_items)]
            filler_indicators.append(filler_indicator_)
        filler_indicators = np.array(filler_indicators)
        # ----------------------
        for t in range(self.T):

            fake_array = fake_profiles * filler_indicators
            # ==================
            fake_array[fake_array < 0] = 0
            fake_array[fake_array > 5] = 5
            fake_array = np.round(fake_array)
            # ==================
            metircs = self.test(victim='SVD', detect=False, fake_array=fake_array)
            print(metircs)
            hitratio_10 = float(metircs.split('\t')[2].split(':')[1])
            # predictions = float(metircs.split('\t')[0].split(':')[1])
            # f_adv_0 = np.sum(predictions)
            f_adv_0 = hitratio_10
            f_adv_k = f_adv_0
            print("opt_adv_intent\tepoch-%d adv goal\t%f" % (t, f_adv_k), flush=True)

            delta_f_Adv = []
            from numpy import linalg as la
            B, Sigma, V = la.svd(fake_profiles)
            for k in range(self.K):

                Z_k = np.matmul(np.reshape(B[k], [self.attack_num, 1]), np.reshape(V[k], [1, self.n_items]))

                fake_users_k = fake_profiles + self.alpha * Z_k

                fake_array = fake_users_k * filler_indicators
                # ==================
                fake_array[fake_array < 0] = 0
                fake_array[fake_array > 5] = 5
                fake_array = np.round(fake_array)
                # ==================
                metircs = self.test(victim='SVD', detect=False, fake_array=fake_array)
                hitratio_10 = float(metircs.split('\t')[2].split(':')[1])
                # predictions = float(metircs.split('\t')[0].split(':')[1])
                f_adv_k_new = hitratio_10

                delta_f_Adv.append((f_adv_k_new - f_adv_k) * Z_k)

            delta_f_A = self.alpha * sum(delta_f_Adv)

            fake_profiles += self.eta * delta_f_A
            fake_profiles[fake_profiles < 0] = 0
            fake_profiles[fake_profiles > 5] = 5
            fake_profiles = np.round(fake_profiles)
        fake_profiles = fake_profiles * filler_indicators
        # fake_profiles[:, self.target_id] = 5.0
        return fake_profiles

    def generate_fakeMatrix(self):
        if self.fake_array is None:
            self.fake_array = self.train()
        return self.fake_array

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
            fake_array = self.train()

            metrics = self.test(victim='all', detect=True, fake_array=fake_array)
            # metrics = self.test(victim='SVD', detect=False, fake_array=fake_array)
            print(metrics, flush=True)
            #
            # self.visualize(log_to_visualize_dict)
            return


    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = self.lrelu(self.conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = self.lrelu(self.d_bn1(self.conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = self.lrelu(self.d_bn2(self.conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = self.lrelu(self.d_bn3(self.conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = self.linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.height, self.width
            # CONV stride=2
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # FC of 2*4*512&ReLU&BN
            self.z_, self.h0_w, self.h0_b = self.linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            # four transposed CONV of [256,128,64] &ReLU&BN&kernel_size = 5 * 5
            self.h1, self.h1_w, self.h1_b = self.deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))
            h2, self.h2_w, self.h2_b = self.deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3, self.h3_w, self.h3_b = self.deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            # transposed CONV of [1] &tanh
            h4, self.h4_w, self.h4_b = self.deconv2d(
                h3, [self.batch_size, s_h, s_w, 1], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    @staticmethod
    def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)  # tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

            return conv

    @staticmethod
    # kernel_size = 5 * 5
    def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            try:
                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                                strides=[1, d_h, d_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
            except:
                deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, biases)  # tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            if with_w:
                return deconv, w, biases
            else:
                return deconv

    @staticmethod
    def lrelu(x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)

    @staticmethod
    def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            try:
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
            except ValueError as err:
                msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
                err.args = err.args + (msg,)
                raise
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias

    @staticmethod
    def conv_out_size_same(size, stride):
        import math
        return int(math.ceil(float(size) / float(stride)))

    @staticmethod
    def gen_random(size):
        # z - N(0,100)
        return np.random.normal(0, 100, size=size)

    @staticmethod
    def conv_cond_concat(x, y):
        """Concatenate conditioning vector on feature map axis."""

        def concat(tensors, axis, *args, **kwargs):
            if "concat_v2" in dir(tf):
                return tf.concat_v2(tensors, axis, *args, **kwargs)
            else:
                return tf.concat(tensors, axis, *args, **kwargs)

        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return concat([
            x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


class DCGANAttacker(WGANAttacker):
    def __init__(self):
        super(DCGANAttacker, self).__init__()

    # ----
    def build_network(self):
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size, self.height, self.width, 1],
                                     name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)

        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.g_loss = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss_real = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        # ===================================
        self.d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        self.g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        # ===================================

    @staticmethod
    def sigmoid_cross_entropy_with_logits(x, y):
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


class HeuristicAttacker(Attacker):
    def __init__(self):
        super(HeuristicAttacker, self).__init__()

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        return parser

    def prepare_data(self):
        super(HeuristicAttacker, self).prepare_data()

    def generate_fakeMatrix(self):
        raise NotImplemented

    def execute(self):
        self.prepare_data()
        res = self.test(victim='all', detect=True)
        # print(res)
        return

    def build_network(self):
        return

    def train(self):
        return

    def save(self, path):
        return

    def restore(self, path):
        return


class RandomAttacker(HeuristicAttacker):
    def __init__(self):
        super(RandomAttacker, self).__init__()

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        # return parser.parse_args()

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(RandomAttacker, self).prepare_data()

        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()
        pass

    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        fake_profiles[:, self.target_id] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id})
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.random.normal(loc=self.global_mean, scale=self.global_std,
                                          size=(self.attack_num * self.filler_num))
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class AverageAttacker(HeuristicAttacker):
    def __init__(self):
        super(AverageAttacker, self).__init__()

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(AverageAttacker, self).prepare_data()

        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()

        self.item_mean_dict = self.train_data_df.groupby('item_id').rating.mean().to_dict()

        self.item_std_dict = self.train_data_df.groupby('item_id').rating.std().fillna(self.global_std).to_dict()
        pass

    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        fake_profiles[:, self.target_id] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id})
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        # sampled_values = np.random.normal(loc=0, scale=1,
        #                                   size=(self.attack_num * self.filler_num))
        sampled_values = [
            np.random.normal(loc=self.item_mean_dict.get(iid, self.global_mean),
                             scale=self.item_std_dict.get(iid, self.global_std))
            for iid in sampled_cols
        ]
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class BandwagonAttacker(HeuristicAttacker):
    def __init__(self):
        super(BandwagonAttacker, self).__init__()
        self.selected_ids = []
        if len(self.args.selected_ids) > 0:
            self.selected_ids = list(map(int, self.args.selected_ids.split(',')))

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        parser.add_argument('--selected_ids', type=str, default='')  # , required=True)

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(BandwagonAttacker, self).prepare_data()

        self.global_mean = self.train_data_df.rating.mean()
        self.global_std = self.train_data_df.rating.std()

        if len(self.selected_ids) == 0:
            sorted_item_pop_df = self.train_data_df.groupby('item_id').agg('count').sort_values('user_id').index[::-1]
            self.selected_ids = sorted_item_pop_df[:1].to_list()
        pass

    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        fake_profiles[:, self.target_id] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id} - set(self.selected_ids))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.random.normal(loc=self.global_mean, scale=self.global_std,
                                          size=(self.attack_num * self.filler_num))
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > 5] = 5
        sampled_values[sampled_values < 1] = 1
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles


class SegmentAttacker(HeuristicAttacker):
    def __init__(self):
        super(SegmentAttacker, self).__init__()
        self.selected_ids = []
        if len(self.args.selected_ids) > 0:
            self.selected_ids = list(map(int, self.args.selected_ids.split(',')))

    @staticmethod
    def parse_args():
        parser = HeuristicAttacker.parse_args()
        parser.add_argument('--selected_ids', type=str, default='')  # , required=True)

        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(SegmentAttacker, self).prepare_data()
        if len(self.selected_ids) == 0:
            import pandas as pd
            p = './data/%s/%s_selected_items' % (self.data_set, self.data_set)
            data = pd.read_csv(p, sep='\t', names=['target_id', 'selected_ids'], engine='python')
            data.target_id = data.target_id.astype(int)
            selected_ids = data[data.target_id == self.target_id].selected_ids.values[0]
            self.selected_ids = list(map(int, str(selected_ids).split(',')))

        # self.global_mean = self.train_data_df.rating.mean()
        # self.global_std = self.train_data_df.rating.std()
        pass

    def generate_fakeMatrix(self):

        fake_profiles = np.zeros(shape=[self.attack_num, self.n_items], dtype=float)
        # padding target score
        fake_profiles[:, self.target_id] = 5
        # padding selected score
        fake_profiles[:, self.selected_ids] = 5
        # padding fillers score
        filler_pool = list(set(range(self.n_items)) - {self.target_id} - set(self.selected_ids))
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        sampled_cols = np.reshape(
            np.array([filler_sampler([filler_pool, self.filler_num]) for _ in range(self.attack_num)]), (-1))
        sampled_rows = [j for i in range(self.attack_num) for j in [i] * self.filler_num]
        sampled_values = np.ones_like(sampled_rows)
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        #
        return fake_profiles
