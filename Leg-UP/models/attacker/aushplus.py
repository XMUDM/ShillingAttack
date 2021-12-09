# -*- coding: utf-8 -*-
# @Time       : 2020/12/3 20:03
# @Author     : chensi
# @File       : legup.py
# @Software   : PyCharm
# @Desciption : None


import random
import numpy as np
import torch
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from datetime import datetime
import time
from torch import nn
import torch.nn.functional as F
import math
import torch.optim as optim

from models.attacker.attacker import Attacker
from models.attacker.aushplus_helper import *
from utils.data_loader import DataLoader


class torchAttacker(Attacker):
    def __init__(self):
        super(torchAttacker, self).__init__()
     
        self.restore_model = self.args.restore_model
   
        self.model_path = self.args.model_path
        
        self.verbose = self.args.verbose
        self.T = self.args.T
      
        self.device = torch.device("cuda:%d" % self.args.cuda_id if self.args.use_cuda > 0 else "cpu")
   
        self.epochs = self.args.epoch
      
        self.lr_G = self.args.lr_G
        self.momentum_G = self.args.momentum_G
    
        self.lr_D = self.args.lr_D
        self.momentum_D = self.args.momentum_D
        self.batch_size_D = self.args.batch_size_D
        self.surrogate = self.args.surrogate
        pass

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--model_path', type=str, default='')
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        #
        parser.add_argument('--use_cuda', type=int, default=1)
        # parser.add_argument('--cuda_id', type=int, default=2)
        parser.add_argument('--epoch', type=int, default=3)
        # Generator
        parser.add_argument("--lr_G", type=float, default=0.01)
        parser.add_argument("--momentum_G", type=float, default=0.99)
        # Discriminator
        parser.add_argument("--lr_D", type=float, default=0.01)
        parser.add_argument("--momentum_D", type=float, default=0.99)
        parser.add_argument('--batch_size_D', type=int, default=64)

        # Surrogate
        parser.add_argument("--epoch_S", type=int, default=50)
        parser.add_argument("--unroll_steps_S", type=int, default=1)
        parser.add_argument("--hidden_dim_S", type=int, default=64)
        parser.add_argument("--lr_S", type=float, default=1e-2)
 
        parser.add_argument("--weight_decay_S", type=float, default=1e-5)
        parser.add_argument('--batch_size_S', type=int, default=64)
       
        parser.add_argument('--weight_pos_S', type=float, default=1.)
        parser.add_argument('--weight_neg_S', type=float, default=0.)
        parser.add_argument("--surrogate", type=str, default="WMF")
        
        gan_args, unknown_args = parser.parse_known_args()
        return gan_args

    def prepare_data(self):

        self.path_train = './data/%s/%s_train.dat' % (self.data_set, self.data_set)
        path_test = './data/%s/%s_test.dat' % (self.data_set, self.data_set)

        dataset_class = DataLoader(self.path_train, path_test)
        self.train_data_df, self.test_data_df, self.n_users, self.n_items = dataset_class.load_file_as_dataFrame()
        train_matrix, _ = dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        test_matrix, _ = dataset_class.dataFrame_to_matrix(self.test_data_df, self.n_users, self.n_items)
        self.train_array, self.test_array = train_matrix.toarray(), test_matrix.toarray()
 
        self.data_loader = torch.utils.data.DataLoader(dataset=torch.from_numpy(self.train_array).type(torch.float32),
                                                       batch_size=self.batch_size_D, shuffle=True, drop_last=True)
 
        self.target_users = np.where(self.train_array[:, self.target_id] == 0)[0]

        attack_target = np.zeros((len(self.target_users), self.n_items))
        attack_target[:, self.target_id] = 1.0
        self.attack_target = torch.from_numpy(attack_target).type(torch.float32).to(self.device)
        pass

    def build_network(self):
        raise NotImplemented


    def get_sur_predictions(self, fake_tensor):
        """
        input：
        fake_tensor：[fake_user_num,item_num]
        surrogate：
        
        output：
        sur_predictions：[user_num+fake_user_num,item_num]
        sur_test_rmse：
        """

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
        data_tensor = torch.cat(
            [torch.from_numpy(self.train_array).type(torch.float32).to(self.device),
             fake_tensor], dim=0)


        surrogate=self.surrogate

        if surrogate == 'WMF':
            sur_trainer_ = WMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dim=self.args.hidden_dim_S,
                device=self.device,
                lr=self.args.lr_S,
                weight_decay=self.args.weight_decay_S,
                batch_size=self.args.batch_size_S,
                weight_pos=self.args.weight_pos_S,
                weight_neg=self.args.weight_neg_S,
                verbose=False)
            epoch_num_ = self.args.epoch_S
            unroll_steps_ = self.args.unroll_steps_S
        elif surrogate == 'ItemAE':
            sur_trainer_ = ItemAETrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dims=self.args.hidden_dim_S,
                device=self.device,
                lr=self.args.lr_S,
                l2=self.args.weight_decay_S,
                batch_size=self.args.batch_size_S,
                weight_pos=self.args.weight_pos_S,
                weight_neg=self.args.weight_neg_S,
                verbose=False)
            epoch_num_ = self.args.epoch_S
            unroll_steps_ = self.args.unroll_steps_S
        elif surrogate == 'SVDpp':
            sur_trainer_ = SVDppTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dims=[128],
                device=self.device,
                lr=1e-3,
                l2=5e-2,
                batch_size=128,
                weight_alpha=20)
            epoch_num_ = 10
            unroll_steps_ = 1
        elif surrogate == 'NMF':
            sur_trainer_ = NMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                batch_size=128,
                device=self.device,
            )
            epoch_num_ = 50
            unroll_steps_ = 1
        elif surrogate == 'PMF':
            sur_trainer_ = PMFTrainer(
                n_users=self.n_users + self.attack_num,
                n_items=self.n_items,
                hidden_dim=128,
                device=self.device,
                lr=0.0001,
                weight_decay=0.1,
                batch_size=self.args.batch_size_S,
                momentum=0.9,
                verbose=True)
            epoch_num_ = 50
            unroll_steps_ = 1
        else:
            print('surrogate model error : ', surrogate)
            exit()

        sur_predictions = sur_trainer_.fit_adv(
            data_tensor=data_tensor,
            epoch_num=epoch_num_,
            unroll_steps=unroll_steps_
        )

        sur_test_rmse = np.mean((sur_predictions[:self.n_users][self.test_array > 0].detach().cpu().numpy()
                                 - self.test_array[self.test_array > 0]) ** 2)
  
        return sur_predictions, sur_test_rmse

    def train_G(self):
        raise NotImplemented

    def save(self, path):


        if path is None or len(path) == 0:
            path = './results/model_saved/%s/%s_%s_%d' % (
                self.data_set, self.data_set, self.__class__.__name__, self.target_id)

        torch.save(self.netG.state_dict(), path + '_G.pkl')
   
        torch.save(self.netD.state_dict(), path + '_D.pkl')
        return

    def restore(self, path):

        if path is None or len(path) == 0:
            path = './results/model_saved/%s/%s_%s_%d' % (
                self.data_set, self.data_set, self.__class__.__name__, self.target_id)

        self.netG.load_state_dict(torch.load(path + '_G.pkl'))
     
        self.netD.load_state_dict(torch.load(path + '_D.pkl'))
        return

    def generate_fakeMatrix(self):

        self.netG.eval()
    
        _, fake_tensor = self.netG(self.real_template)
      
        fake_tensor[:, self.target_id] = 5
       
        return fake_tensor.detach().cpu().numpy()

    @staticmethod
    def custimized_attack_loss(logits, labels):

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -log_probs * labels
        instance_data = labels.sum(1)
        instance_loss = loss.sum(1)
        # Avoid divide by zeros.
        res = instance_loss / (instance_data + 0.1)  # PSILON)
        return res

    @staticmethod
    def update_params(loss, optimizer):

        grad_groups = torch.autograd.grad(loss.cuda(), [x.cuda() for x in optimizer.param_groups[0]['params']], allow_unused=True)
  
        for para_, grad_ in zip(optimizer.param_groups[0]['params'], grad_groups):
            if para_.grad is None:
                para_.grad = grad_
            else:
                para_.grad.data = grad_

        optimizer.step()
        pass
        #for name, param in optimizer.named_parameters()

"""
Revisiting Adversarially Learned Injection Attacks Against Recommender Systems, 
Jiaxi Tang, Hongyi Wen and Ke Wang , RecSys '20
"""
class AIA(torchAttacker):
    def __init__(self):
        super(AIA, self).__init__()
        # ml100k:lr=0.5,epoch=10
        pass

    def build_network(self):

        sampled_idx = np.random.choice(np.where(np.sum(self.train_array > 0, 1) >= self.filler_num)[0],
                                       self.attack_num)
        templates = self.train_array[sampled_idx]

        for (idx, template) in enumerate(templates):
            fillers = np.where(template)[0]
            np.random.shuffle(fillers)
            for iid in fillers[self.filler_num:]:
                templates[idx][iid] = 0.
      
        self.real_template = torch.tensor(templates).type(torch.float).to(self.device)
   
        self.netG = RecsysGenerator(self.device, self.real_template).to(self.device)
      
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)
        pass

    def train_G(self):

        self.netG.train()

        fake_tensor_distribution, fake_tensor = self.netG(self.real_template)
      
        sur_predictions, sur_test_rmse = self.get_sur_predictions(fake_tensor)
        higher_mask = (sur_predictions[self.target_users] >=
                       (sur_predictions[self.target_users, self.target_id].reshape([-1, 1]))).float()
        G_loss = self.custimized_attack_loss(logits=sur_predictions[self.target_users] * higher_mask,
                                             labels=self.attack_target).mean()

        self.update_params(G_loss, self.G_optimizer)
        self.netG.eval()
      
        return G_loss.item()

    def train(self):

        log_to_visualize_dict = {}

        for epoch in range(self.epochs):
            if self.verbose and epoch % self.T == 0:
                datetime_begin = datetime.now()

            G_loss = self.train_G()
            if self.verbose and epoch % self.T == 0:
                train_time = (datetime.now() - datetime_begin).seconds

                metrics = self.test(victim='SVD', detect=False)

                print("epoch:%d\ttime:%ds" % (epoch, train_time), end='\t')
                print("G_loss:%.4f" % (G_loss), end='\t')
                print(metrics, flush=True)

        return log_to_visualize_dict
        pass

    def execute(self):

        self.prepare_data()

        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            log_to_visualize_dict = self.train()
            print("training done.")

            # self.save(self.model_path)

        metrics = self.test(victim='all', detect=True)
        print(metrics, flush=True)
        return


class AUSHplus(torchAttacker):
    def __init__(self):
        super(AUSHplus, self).__init__()
        print('Args:\n', self.args, '\n', flush=True)
        pass

    def build_network(self):

        sampled_idx = np.random.choice(range(self.n_users), self.attack_num)
        templates = self.train_array[sampled_idx]

        for (idx, template) in enumerate(templates):
            fillers = np.where(template)[0]
            np.random.shuffle(fillers)
            for iid in fillers[self.filler_num:]:
                templates[idx][iid] = 0.

        self.real_template = torch.tensor(templates).type(torch.float).to(self.device)

        self.netG = DiscretGenerator_AE_1(self.device, p_dims=[self.n_items, 125]).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)

        self.netD = Discriminator(input_dim=self.n_items).to(self.device)
        self.D_optimizer = optim.Adam(self.netD.parameters(), lr=self.lr_D)
        pass

    def train_D(self):

        self.netD.train()

        _, fake_tensor = self.netG(self.real_template)
        fake_tensor = fake_tensor.detach()

        D_loss_list = []
        for real_tensor in self.data_loader:

            real_tensor = real_tensor.to(self.device)[:self.attack_num]
            # forward
            self.D_optimizer.zero_grad()
            D_real = self.netD(real_tensor)
            D_fake = self.netD(fake_tensor)
            # loss
            D_real_loss = nn.BCELoss()(D_real,
                                       torch.ones_like(D_real).to(self.device))
            D_fake_loss = nn.BCELoss()(D_fake,
                                       torch.zeros_like(D_fake).to(self.device))
            D_loss = D_real_loss + D_fake_loss
            # backward
            D_loss.backward()
            self.D_optimizer.step()

            D_loss_list.append(D_loss.item())
            #
            # break
        self.netD.eval()

        return np.mean(D_loss_list)

    def train_G(self, adv=True, attack=True):

        self.netG.train()

        fake_tensor_distribution, fake_tensor = self.netG(self.real_template)

        G_adv_loss = torch.tensor(0.)
        if adv:
            G_adv_loss = nn.BCELoss(reduction='mean')(self.netD(fake_tensor),
                                                      torch.ones(fake_tensor.shape[0], 1).to(self.device))
            # """crossEntropy loss"""
            # real_labels_flatten = self.real_template.flatten().type(torch.long)
            # fake_logits_flatten = fake_tensor_distribution.reshape([-1, 5])
            # G_rec_loss = nn.CrossEntropyLoss()(fake_logits_flatten[real_labels_flatten > 0],
            #                                    real_labels_flatten[real_labels_flatten > 0] - 1)
            G_adv_loss = G_adv_loss  # + G_rec_loss

        real_labels_flatten = self.real_template.flatten().type(torch.long)
        MSELoss = nn.MSELoss()(fake_tensor.flatten()[real_labels_flatten > 0],
                               real_labels_flatten[real_labels_flatten > 0])

        G_attack_loss = torch.tensor(0.)
        if attack:
            sur_predictions, sur_test_rmse = self.get_sur_predictions(fake_tensor)
            higher_mask = (sur_predictions[self.target_users] >=
                           (sur_predictions[self.target_users, self.target_id].reshape([-1, 1]))).float()
            G_attack_loss = self.custimized_attack_loss(logits=sur_predictions[self.target_users] * higher_mask,
                                                        labels=self.attack_target).mean()

        G_loss = G_adv_loss + G_attack_loss

        self.update_params(G_loss, self.G_optimizer)
        self.netG.eval()

        mean_score = fake_tensor[fake_tensor > 0].mean().item()
        return (G_loss.item(), MSELoss.item(), G_attack_loss.item(), mean_score)

    def pretrain_G(self):

        self.netG.train()
        G_loss_list = []
        for real_tensor in self.data_loader:
            # input data
            real_tensor = real_tensor.to(self.device)
            # forward
            fake_tensor_distribution, fake_tensor = self.netG(real_tensor)
            # crossEntropy loss
            real_labels_flatten = real_tensor.flatten().type(torch.long)
            fake_logits_flatten = fake_tensor_distribution.reshape([-1, 5])
            G_rec_loss = nn.CrossEntropyLoss()(fake_logits_flatten[real_labels_flatten > 0],
                                               real_labels_flatten[real_labels_flatten > 0] - 1)
            G_loss = G_rec_loss
            MSELoss = nn.MSELoss()(fake_tensor.flatten()[real_labels_flatten > 0],
                                   real_labels_flatten[real_labels_flatten > 0])
            #backword
            self.update_params(G_loss.cuda(), self.G_optimizer)
            G_loss_list.append(G_loss.item())
        self.netG.eval()
        return (np.mean(G_loss_list), MSELoss)

    def train(self):

        log_to_visualize_dict = {}

        print('======================pretrain begin======================')
        print('pretrain G......')
        pretrain_epoch = 1 if self.data_set == 'automotive' else 15
        for i in range(pretrain_epoch):
            G_loss, MSELoss = self.pretrain_G()
            if i % 5 == 0:
                print("G_loss:%.4f\tMSELoss:%.4f" % (G_loss, MSELoss))

        print('pretrain D......')
        for i in range(5):
            D_loss = self.train_D()
            print("D_loss:%.4f" % (D_loss))
        print('======================pretrain end======================\n')

        for epoch in range(self.epochs):
            print('==============epoch%d===============' % epoch)

            for epoch_gan_d in range(5):
                D_loss = self.train_D()
                print("D_loss:%.4f" % (D_loss))

            for epoch_gan_g in range(1):  # 5/1
                _, _, G_adv_loss, _ = self.train_G(adv=True, attack=False)
                _, fake_tensor = self.netG(self.real_template)
                # metrics = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                # metrics = "%.4f" % metrics
                # print(metrics)

            for epoch_surrogate in range(50):
                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_begin = datetime.now()

                _, MSELoss, G_rec_loss, mean_score = self.train_G(adv=False, attack=True)
                G_loss = G_adv_loss + G_rec_loss

                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_end = datetime.now()
                    train_time = (datetime_end - datetime_begin).seconds
                    metrics = ''
                    _, fake_tensor = self.netG(self.real_template)
                    metrics2 = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                    metrics2 = ("%.4f" % metrics2)
                    # 打印日志
                    print("epoch:%d-%d\ttime:%ds" % (epoch, epoch_surrogate, train_time), end='\t')
                    print("D_loss:%.4f\tG_loss:%.4f" % (D_loss, G_loss), end='\t')
                    print(metrics, metrics2, flush=True)

            metrics = self.test(victim='SVD', detect=False)
            print(metrics, flush=True)

        return log_to_visualize_dict
        pass

    def execute(self):

        self.prepare_data()
        # Build and compile GAN Network
        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            log_to_visualize_dict = self.train()
            print("training done.")

            # self.save(self.model_path)

        metrics = self.test(victim='all', detect=True)
        print(metrics, flush=True)
        return


class AUSHplus_SR(AUSHplus):
    def __init__(self):
        super(AUSHplus_SR, self).__init__()
        pass

    def build_network(self):
        super(AUSHplus_SR, self).build_network()
        self.netG = RoundGenerator_AE_1(self.device, p_dims=[self.n_items, 125]).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)
        pass

    def pretrain_G(self):
        self.netG.train()
        G_loss_list = []
        for real_tensor in self.data_loader:
            # input data
            real_tensor = real_tensor.to(self.device)
            # forward
            _, fake_tensor = self.netG(real_tensor)
            """MSELoss"""
            real_labels_flatten = real_tensor.flatten()
            MSELoss = nn.MSELoss()(fake_tensor.flatten()[real_labels_flatten > 0],
                                   real_labels_flatten[real_labels_flatten > 0])
            G_loss = MSELoss
            """train"""
            self.update_params(G_loss, self.G_optimizer)
            G_loss_list.append(G_loss.item())

        self.netG.eval()
        return (np.mean(G_loss_list), MSELoss)


class AUSHplus_woD(AUSHplus):
    def __init__(self):
        super(AUSHplus_woD, self).__init__()
        pass

    def train(self):
        log_to_visualize_dict = {}
        print('======================pretrain begin======================')
        print('pretrain G......')
        for i in range(1):  # 15
            G_loss, MSELoss = self.pretrain_G()
            if i % 5 == 0:
                print("G_loss:%.4f\tMSELoss:%.4f" % (G_loss, MSELoss))
        # print('pretrain D......')
        # for i in range(5):
        #     D_loss = self.train_D()
        #     print("D_loss:%.4f" % (D_loss))
        print('======================pretrain end======================\n')
        for epoch in range(self.epochs):
            print('==============epoch%d===============' % epoch)
            # for epoch_gan_d in range(5):
            #     D_loss = self.train_D()
            #     print("D_loss:%.4f" % (D_loss))
            # for epoch_gan_g in range(1):  # 5
            #     _, _, G_adv_loss, _ = self.train_G(adv=True, attack=False)
            # =============================
            for epoch_surrogate in range(100):
                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_begin = datetime.now()
                # ================================
                _, MSELoss, G_rec_loss, mean_score = self.train_G(adv=False, attack=True)
                G_loss = G_rec_loss  # G_adv_loss + G_rec_loss
                # ================================

                if self.verbose and epoch_surrogate % self.T == 0:
                    # =============================
                    datetime_end = datetime.now()
                    train_time = (datetime_end - datetime_begin).seconds
                    # =============================
                    metrics = ''
                    # if epoch_surrogate % 10 == 0:
                    #     metrics = self.test(victim='NeuMF', detect=False)
                    _, fake_tensor = self.netG(self.real_template)
                    metrics2 = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                    metrics2 = ("%.4f" % metrics2)
                    #
                    print("epoch:%d-%d\ttime:%ds" % (epoch, epoch_surrogate, train_time), end='\t')
                    print("G_loss:%.4f" % (G_loss), end='\t')
                    print(metrics, metrics2, flush=True)
                # =============================
            metrics = self.test(victim='SVD', detect=False)
            print(metrics, flush=True)
        # print(self.test(victim='all', detect=True))
        # exit()
        return log_to_visualize_dict
        pass


class AUSHplus_SF(AUSHplus):
    def __init__(self):
        super(AUSHplus_SF, self).__init__()
        pass

    def build_network(self):
        super(AUSHplus_SF, self).build_network()
        self.netG = DiscretRecsysGenerator_1(self.device, self.real_template).to(self.device)
        self.G_optimizer = optim.Adam(self.netG.parameters(), lr=self.lr_G)
        pass

    def train(self):
        log_to_visualize_dict = {}
        print('======================pretrain begin======================')
        # print('pretrain G......')
        # for i in range(15):
        #     G_loss, MSELoss = self.pretrain_G()
        #     if i % 5 == 0:
        #         print("G_loss:%.4f\tMSELoss:%.4f" % (G_loss, MSELoss))
        print('pretrain D......')
        for i in range(5):
            D_loss = self.train_D()
            print("D_loss:%.4f" % (D_loss))
        print('======================pretrain end======================\n')
        for epoch in range(self.epochs):
            print('==============epoch%d===============' % epoch)
            for epoch_gan_d in range(5):
                D_loss = self.train_D()
                print("D_loss:%.4f" % (D_loss))
            for epoch_gan_g in range(1):  # 5
                _, _, G_adv_loss, _ = self.train_G(adv=True, attack=False)
            # =============================
            for epoch_surrogate in range(100):
                if self.verbose and epoch_surrogate % self.T == 0:
                    datetime_begin = datetime.now()
                # ================================
                _, MSELoss, G_rec_loss, mean_score = self.train_G(adv=False, attack=True)
                G_loss = G_adv_loss + G_rec_loss
                # ================================

                if self.verbose and epoch_surrogate % self.T == 0:
                    # =============================
                    datetime_end = datetime.now()
                    train_time = (datetime_end - datetime_begin).seconds
                    # =============================
                    metrics = ''
                    _, fake_tensor = self.netG(self.real_template)
                    metrics2 = fake_tensor.detach()[fake_tensor.detach() > 0].mean().item()
                    metrics2 = ("%.4f" % metrics2)
                    #
                    print("epoch:%d-%d\ttime:%ds" % (epoch, epoch_surrogate, train_time), end='\t')
                    print("D_loss:%.4f\tG_loss:%.4f" % (D_loss, G_loss), end='\t')
                    print(metrics, metrics2, flush=True)
        # print(self.test(victim='all', detect=True))
        # exit()
        return log_to_visualize_dict
        pass


class AUSHplus_inseg(AUSHplus):
    def __init__(self):
        super(AUSHplus_inseg, self).__init__()
        print(self.test())
        exit()
        pass

    def prepare_data(self):
        super(AUSHplus_inseg, self).prepare_data()
        #
        path = './data/%s/%s_target_users' % (self.data_set, self.data_set)
        with open(path) as lines:
            for line in lines:
                if int(line.split('\t')[0]) == self.target_id:
                    self.target_users = list(map(int, line.split('\t')[1].split(',')))
                    target_users = list(self.target_users) + list(map(int, line.split('\t')[1].split(','))) * 4
                    self.target_users = np.array(target_users)
                    break
        # self.target_users = np.where(self.train_array[:, self.target_id] == 0)[0]
        attack_target = np.zeros((len(self.target_users), self.n_items))
        attack_target[:, self.target_id] = 1.0
        self.attack_target = torch.from_numpy(attack_target).type(torch.float32).to(self.device)
        pass

    def test(self, victim='SVD', detect=False, fake_array=None):
        """

        :param victim:
        :param evalutor:
        :return:
        """

        # self.generate_injectedFile(fake_array)
        """detect"""
        # res_detect_list = self.detect(detect)
        # res_detect = '\t'.join(res_detect_list)

        """attack"""
        # if self.target_id == 594:
        #     all_victim_models = ['NeuMF', 'IAutoRec']
        # if self.target_id in [1257]:
        #     all_victim_models = ['NeuMF', 'IAutoRec', 'UAutoRec']
        if self.target_id in [1077]:
            all_victim_models = ['NeuMF']
        # if victim is False:
        #     res_attack = ''
        # elif victim in all_victim_models:
        #     """攻击victim model"""
        #     self.attack(victim)
        #     #
        #     """对比结果"""
        #     res_attack_list = self.evaluate(victim)
        #     res_attack = '\t'.join(res_attack_list)
        #     #
        # else:
        #     if victim == 'all':
        if True:
            victim_models = all_victim_models
            # else:
            #     victim_models = victim.split(',')
            res_attack_list = []
            # SlopeOne,SVD,NMF,IAutoRec,UAutoRec,NeuMF
            for victim_model in victim_models:
                """攻击victim model"""
                self.attack(victim_model)
                #
                """对比结果"""
                cur_res_list = self.evaluate(victim_model)
                #
                res_attack_list.append('\t:\t'.join([victim_model, '\t'.join(cur_res_list)]))
            res_attack = '\n' + '\n'.join(res_attack_list)
        res = '\t'.join([res_attack])
        return res
