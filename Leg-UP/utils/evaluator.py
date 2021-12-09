# -*- coding: utf-8 -*-
# @Time       : 2020/11/29 18:41
# @Author     : chensi
# @File       : evaluator.py
# @Software   : PyCharm
# @Desciption : None
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
import argparse
from utils.data_loader import DataLoader
import numpy as np, pandas as pd
import scipy.stats
import random
import time, os
from models.detector.SDLib.main.SDLib import SDLib
from models.detector.SDLib.tool.config import Config


class Evaluator(object):
    def __init__(self):
        self.args = self.parse_args()
        self.data_path_clean = self.args.data_path_clean
        self.data_path_attacked = self.args.data_path_attacked
        pass

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Run Evaluator.")
        parser.add_argument('--data_path_clean', type=str,
                            default='./data/filmTrust/filmTrust_train.dat')
        # default='./results/performance/mid_results/ml100k/ml100k_SVD_62.npy')  # , required=True)
        parser.add_argument('--data_path_attacked', type=str,
                            default='./results/data_attacked/filmTrust/filmTrust_AUSH_5.data')
        # default='./results/performance/mid_results/ml100k/ml100k_SVD_aushplus_62.npy')  # , required=True)
        return parser

    def execute(self):
        raise NotImplementedError


class Attack_Effect_Evaluator(Evaluator):
    def __init__(self):
        super(Attack_Effect_Evaluator, self).__init__()
        self.topk_list = list(map(int, self.args.topk.split(',')))
        print("Attack_Effect_Evaluator.")

    @staticmethod
    def parse_args():
        parser = Evaluator.parse_args()
        parser.add_argument('--topk', type=str, default='5,10,20,50')
        parser.add_argument('--target_users', type=str, default='all')
        args, _ = parser.parse_known_args()
        return args

    def execute(self):
        #
        predResults_target = np.load(self.data_path_clean)
        predResults_target = pd.DataFrame(predResults_target)
        predResults_target.columns = ['user_id', 'rating'] + ['hr_%d' % i for i in self.topk_list]
        predResults_target.user_id = predResults_target.user_id.astype(int)
        #
        predResults_target_attacked = np.load(self.data_path_attacked)
        predResults_target_attacked = pd.DataFrame(predResults_target_attacked)
        predResults_target_attacked.columns = ['user_id', 'rating_attacked'] + ['hr_%d_attacked' % i for i in
                                                                                self.topk_list]
        predResults_target_attacked.user_id = predResults_target_attacked.user_id.astype(int)
        #
        print('\n' * 3)
        print(self.data_path_attacked)
        print(predResults_target_attacked.mean())
        print('\n' * 3)
        #
        if self.args.target_users != 'all':
            target_users = list(map(int, self.args.target_users.split(',')))
            predResults_target = predResults_target[predResults_target.user_id.apply(lambda x: x in target_users)]
            predResults_target_attacked = predResults_target_attacked[
                predResults_target_attacked.user_id.apply(lambda x: x in target_users)]

        #
        result = pd.merge(predResults_target, predResults_target_attacked, on=['user_id'])
        result['pred_shift'] = result['rating_attacked'] - result['rating']
        #
        keys = ['pred_shift'] + ['hr_%d_attacked' % i for i in self.topk_list]
        result = result.mean()[keys]
        # res_str = '%.4f\t' * 5 % tuple(result.values)
        res_str = '\t'.join(["%s:%.4f" % (k.replace('_attacked', ''), result[k]) for k in keys])
        print('result begin', res_str, 'result end')
        return res_str


class Profile_Distance_Evaluator(Evaluator):
    def __init__(self):
        super(Profile_Distance_Evaluator, self).__init__()
        print("Profile_Distance_Evaluator.")

    @staticmethod
    def parse_args():
        parser = Evaluator.parse_args()
        args, _ = parser.parse_known_args()
        return args

    @staticmethod
    def get_TVD_distance(P, Q):
        dis_TVD = np.mean(np.sum(np.abs(P - Q) / 2, 1))
        return dis_TVD

    @staticmethod
    def get_JS_distance(P, Q):
        fn_KL = lambda p, q: scipy.stats.entropy(p, q)
        M = (P + Q) / 2
        js_vec = []
        for iid in range(P.shape[0]):
            p, q, m = P[iid], Q[iid], M[iid]
            js_vec.append((fn_KL(p, m) + fn_KL(q, m)) / 2)
        # drop nan
        dis_JS = np.mean([i for i in js_vec if not np.isnan(i)])
        return dis_JS

    @staticmethod
    # cacu item distribution
    def get_item_distribution(profiles):
        profiles_T = profiles.transpose()
        item_rating_distribution = np.concatenate([np.expand_dims(np.sum(profiles_T == i, 1), 0)
                                                   for i in range(6)]).transpose()
        # normalize
        return item_rating_distribution / profiles.shape[0]

    def execute(self):
        # self.data_path_clean = './data/ml100k/ml100k_train.dat'
        # self.data_path_attacked = './results/data_attacked/ml100k/ml100k_AUSH_0.data'
        #
        path_test = self.data_path_clean.replace('train', 'test')
        # load real profile matrix
        dataset_class_real = DataLoader(self.data_path_clean, path_test)
        train_data_df_real, _, n_users_real, n_items_real = dataset_class_real.load_file_as_dataFrame()
        train_matrix_real, _ = dataset_class_real.dataFrame_to_matrix(train_data_df_real, n_users_real, n_items_real)
        train_matrix_real = train_matrix_real.toarray()

        # load fake profile matrix
        dataset_class_attacked = DataLoader(self.data_path_attacked, path_test)
        train_data_df_attacked, _, n_users_attacked, n_items_attacked = dataset_class_attacked.load_file_as_dataFrame()
        train_matrix_attacked, _ = dataset_class_attacked.dataFrame_to_matrix(train_data_df_attacked, n_users_attacked,
                                                                              n_items_attacked)
        train_matrix_fake = train_matrix_attacked.toarray()[n_users_real:, :]

        # cacu item distribution
        real_item_distribution = self.get_item_distribution(train_matrix_real)
        fake_item_distribution = self.get_item_distribution(train_matrix_fake)
        #
        TVD_distance = self.get_TVD_distance(real_item_distribution, fake_item_distribution)
        JS_distance = self.get_JS_distance(real_item_distribution, fake_item_distribution)
        #
        res_str = 'TVD:%.4f\tJS:%.4f' % (TVD_distance, JS_distance)
        print('result begin', res_str, 'result end')
        return TVD_distance, JS_distance


class FAP_Detector(Evaluator):
    def __init__(self):
        super(FAP_Detector, self).__init__()
        print("FAP_ShillingAttack_Detector.")

    @staticmethod
    def parse_args():
        parser = Evaluator.parse_args()
        args, _ = parser.parse_known_args()
        return args

    def execute(self):
        # temp file path
        cur_time = time.time()
        label_file_path = './label_%f.tmp' % cur_time
        conf_file_path = './conf_%f.tmp' % cur_time

        args = {
            'ratings': self.data_path_attacked,
            'ratings.setup': '-columns 0 1 2',
            'label': label_file_path,
            'methodName': 'FAP',
            'evaluation.setup': '-ap 0.000001',
            'seedUser': 5,
            'topKSpam': 50,
            'output.setup': 'on -dir ./',
        }

        # write conf file
        with open(conf_file_path, 'w') as fout:
            fout.write('\n'.join(['%s=%s' % i for i in args.items()]))

        # write label file
        _, _, n_users_real, _ = DataLoader(self.data_path_clean,
                                           self.data_path_clean.replace('train', 'test'),
                                           verbose=False).load_file_as_dataFrame()
        _, _, n_users_attacked, _ = DataLoader(self.data_path_attacked,
                                               self.data_path_clean.replace('train', 'test'),
                                               verbose=False).load_file_as_dataFrame()

        uids, labels = np.arange(n_users_attacked), np.zeros(n_users_attacked)
        labels[n_users_real:] = 1

        with open(label_file_path, 'w') as fout:
            fout.write('\n'.join(["%d\t%d" % i for i in list(zip(uids, labels))]))

        sd = SDLib(Config(conf_file_path))
        result = sd.execute()
        res_str = "pre:%.4f\trecall:%.4f" % tuple(result)
        print('result begin', res_str, 'result end')
        #
        os.remove(label_file_path)
        os.remove(conf_file_path)
        #
        pass


class Tsne_Evaluator(Evaluator):
    def __init__(self):
        self.args = self.parse_args()
        self.data_set = self.args.data_set
        self.target_id = self.args.target_id
        self.attacker = self.args.attacker
        self.recommender = self.args.recommender
        #
        self.no_dims = self.args.no_dims
        self.perplexity = self.args.perplexity
        self.max_iter = self.args.max_iter
        self.initial_momentum = self.args.initial_momentum
        self.final_momentum = self.args.final_momentum
        self.eta = self.args.eta
        self.min_gain = self.args.min_gain
        self.tol = self.args.tol

        print("Tsne_Evaluator.")

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Run Evaluator.")
        #
        parser.add_argument('--data_set', type=str, default='ml100k')
        parser.add_argument('--target_id', type=int, default=62)
        #         attackers = ['AUSHplus_Dis_xiaorong', 'AUSHplus', 'SegmentAttacker', 'BandwagonAttacker',
        #                      'AverageAttacker', 'RandomAttacker',
        #                      'AUSH', 'RecsysAttacker',
        #                      'DCGAN', 'WGAN']
        parser.add_argument('--attacker', type=str, default='SegmentAttacker')  # AUSHplus,RecsysAttacker
        parser.add_argument('--recommender', type=str, default='NeuMF')
        #
        parser.add_argument('--no_dims', type=int, default=2)
        parser.add_argument('--perplexity', type=int, default=30.0)
        parser.add_argument('--max_iter', type=int, default=1000)
        parser.add_argument('--initial_momentum', type=float, default=0.5)
        parser.add_argument('--final_momentum', type=float, default=0.8)
        parser.add_argument('--eta', type=int, default=500)
        parser.add_argument('--min_gain', type=float, default=0.01)
        parser.add_argument('--tol', type=float, default=1e-5)
        #
        args, _ = parser.parse_known_args()
        return args

    def execute(self):

        import numpy as np
        import matplotlib.pyplot as plt
        import time
        # load data
        # =================================
        # path_dir = './results/performance/mid_results/%s' % (self.data_set)
        # user_embed_path = '%s/%s_NeuMF_%s_%d_user_embed.npy' % (
        #     path_dir, self.data_set, self.attacker, self.target_id)
        #
        # self.x = np.load(user_embed_path)
        # #

        # =================================
        train_path = './results/data_attacked/%s/%s_%s_%d.data' % (
            self.data_set, self.data_set, self.attacker, self.target_id)
        test_path = './data/%s/%s_test.dat' % (self.data_set, self.data_set)
        dataset_class_attacked = DataLoader(train_path, test_path)
        train_data_df_attacked, _, n_users_attacked, n_items_attacked = dataset_class_attacked.load_file_as_dataFrame()
        train_matrix_attacked, _ = dataset_class_attacked.dataFrame_to_matrix(train_data_df_attacked, n_users_attacked,
                                                                              n_items_attacked)
        self.x = train_matrix_attacked.toarray()
        # =================================
        Y = np.ones(self.x.shape[0])
        Y[-50:] = 0
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(self.x)
        data_2d = pca.transform(self.x)
        # plt.scatter(data_2d[:, 0], data_2d[:, 1], c=Y)
        # # plt.show()
        # # exit()
        # fig_path = "./results/performance/figs/%s/Tsne_%s_%s_%d_profile_pca.png" \
        #            % (self.data_set, self.attacker, self.recommender, self.target_id)
        # plt.savefig(fig_path)
        data_path = "./results/performance/figs/%s/Tsne_%s_%s_%d_profile_pca" \
                    % (self.data_set, self.attacker, self.recommender, self.target_id)
        np.save(data_path, data_2d)
        exit()
        # ==================================
        # ==================================
        Y = np.ones(self.x.shape[0])
        Y[-50:] = 0
        #
        (n, d) = self.x.shape
        # 随机初始化Y
        y = np.random.randn(n, self.no_dims)
        # dy梯度
        dy = np.zeros((n, self.no_dims))
        # iy是什么
        iy = np.zeros((n, self.no_dims))

        gains = np.ones((n, self.no_dims))
        # 对称化
        P = self.seach_prob()
        P = P + np.transpose(P)
        P = P / np.sum(P)  # pij
        # early exaggeration
        # pi\j
        print("T-SNE DURING:%s" % time.clock())
        P = P * 4
        P = np.maximum(P, 1e-12)
        # Run iterations
        for iter in range(self.max_iter):
            # Compute pairwise affinities
            sum_y = np.sum(np.square(y), 1)
            num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
            num[range(n), range(n)] = 0
            Q = num / np.sum(num)  # qij
            Q = np.maximum(Q, 1e-12)

            # Compute gradient
            # np.tile(A,N)  [1],5 [1,1,1,1,1]
            # pij-qij
            PQ = P - Q

            for i in range(n):
                dy[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.no_dims, 1)).T * (y[i, :] - y), 0)

            # Perform the update
            if iter < 20:
                momentum = self.initial_momentum
            else:
                momentum = self.final_momentum

            gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
            gains[gains < self.min_gain] = self.min_gain

            iy = momentum * iy - self.eta * (gains * dy)
            y = y + iy
            y = y - np.tile(np.mean(y, 0), (n, 1))
            # Compute current value of cost function\
            if (iter + 1) % 100 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Iteration ", (iter + 1), ": error is ", C)
                if (iter + 1) != 100:
                    ratio = C / oldC
                    print("ratio ", ratio)
                    if ratio >= 0.95:
                        break
                oldC = C
            # Stop lying about P-values
            if iter == 100:
                P = P / 4
        print("finished training!")
        #
        data_2d = y
        # plt.scatter(data_2d[:, 0], data_2d[:, 1], c=Y)
        # plt.show()
        # fig_path = "./results/performance/figs/%s/Tsne_%s_%s_%d.png" \
        #            % (self.data_set, self.attacker, self.recommender, self.target_id)
        # plt.savefig(fig_path)
        data_path = "./results/performance/figs/%s/Tsne_%s_%s_%d_profile" \
                    % (self.data_set, self.attacker, self.recommender, self.target_id)
        np.save(data_path, data_2d)
        pass

    def seach_prob(self):

        print("Computing pairwise distances...")
        (n, d) = self.x.shape
        dist = self.cal_pairwise_dist()
        dist[dist < 0] = 0
        pair_prob = np.zeros((n, n))
        beta = np.ones((n, 1))

        base_perp = np.log(self.perplexity)

        for i in range(n):
            if i % 500 == 0:
                print("Computing pair_prob for point %s of %s ..." % (i, n))

            betamin = -np.inf
            betamax = np.inf
            perp, this_prob = self.cal_perplexity(dist[i], i, beta[i])

            perp_diff = perp - base_perp
            tries = 0
            while np.abs(perp_diff) > self.tol and tries < 50:
                if perp_diff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2

                perp, this_prob = self.cal_perplexity(dist[i], i, beta[i])
                perp_diff = perp - base_perp
                tries = tries + 1

            pair_prob[i,] = this_prob
        print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))

        return pair_prob

    def cal_pairwise_dist(self):
        sum_x = np.sum(np.square(self.x), 1)
        dist = np.add(np.add(-2 * np.dot(self.x, self.x.T), sum_x).T, sum_x)

        return dist

    def cal_perplexity(self, dist, idx=0, beta=1.0):

        prob = np.exp(-dist * beta)

        prob[idx] = 0
        sum_prob = np.sum(prob)
        if sum_prob < 1e-12:
            prob = np.maximum(prob, 1e-12)
            perp = -12
        else:
            perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
            prob /= sum_prob

        return perp, prob
