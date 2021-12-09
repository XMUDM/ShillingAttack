# -*- coding: utf-8 -*-
# @Time       : 2020/12/6 12:54
# @Author     : chensi
# @File       : AUSH.py
# @Software   : PyCharm
# @Desciption : None

import random
import numpy as np
import torch

import tensorflow as tf

seed = 1234
random.seed(seed)
np.random.seed(seed)
# tf.set_random_seed(seed)
tf.random.set_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import math, scipy
import keras
from keras.layers import Input, Dense, Lambda, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2 as L2

from models.attacker.attacker import Attacker


class AUSH(Attacker):

    def __init__(self):
        super(AUSH, self).__init__()
        self.selected_ids = list(map(int, self.args.selected_ids.split(',')))
        #
        self.restore_model = self.args.restore_model
        self.model_path = self.args.model_path
        #
        self.epochs = self.args.epoch
        self.batch_size = self.args.batch_size
        #
        self.learning_rate_G = self.args.learning_rate_G
        self.reg_rate_G = self.args.reg_rate_G
        self.ZR_ratio = self.args.ZR_ratio
        #
        self.learning_rate_D = self.args.learning_rate_D
        self.reg_rate_D = self.args.reg_rate_D
        #
        self.verbose = self.args.verbose
        self.T = self.args.T
        #

    @staticmethod
    def parse_args():
        parser = Attacker.parse_args()
        parser.add_argument('--selected_ids', type=str, default='1,2,3', required=True)
        #
        parser.add_argument('--restore_model', type=int, default=0)
        parser.add_argument('--model_path', type=str, default='')
        #
        parser.add_argument('--epoch', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=256)
        #
        parser.add_argument('--learning_rate_G', type=float, default=0.01)
        parser.add_argument('--reg_rate_G', type=float, default=0.0001)
        parser.add_argument('--ZR_ratio', type=float, default=0.2)
        #
        parser.add_argument('--learning_rate_D', type=float, default=0.001)
        parser.add_argument('--reg_rate_D', type=float, default=1e-5)
        #
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument('--T', type=int, default=5)
        #
        args, _ = parser.parse_known_args()
        return args

    def prepare_data(self):
        super(AUSH, self).prepare_data()
        train_matrix, _ = self.dataset_class.dataFrame_to_matrix(self.train_data_df, self.n_users, self.n_items)
        self.train_data_array = train_matrix.toarray()
        self.train_data_mask_array = scipy.sign(self.train_data_array)

        mask_array = (self.train_data_array > 0).astype(np.float)
        mask_array[:, self.selected_ids + [self.target_id]] = 0
        self.template_idxs = np.where(np.sum(mask_array, 1) >= self.filler_num)[0]

    def build_network(self):
        optimizer_G = Adam(learning_rate=self.learning_rate_G)
        optimizer_D = Adam(learning_rate=self.learning_rate_D)

        ########################################
        # Build and compile the discriminator
        ########################################
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_D,
                                   metrics=['accuracy'])

        ########################################
        # Build the generator
        ########################################
        self.generator = self.build_generator()
        # Inputs of generator
        real_profiles = Input(shape=(self.n_items,))
        fillers_mask = Input(shape=(self.n_items,))
        selects_mask = Input(shape=(self.n_items,))
        target_patch = Input(shape=(self.n_items,))
        # Outputs of generator
        fake_profiles = self.generator([real_profiles, fillers_mask, selects_mask, target_patch])
        ########################################
        # For the combined model we will only train the generator
        ########################################
        self.discriminator.trainable = False
        # discriminator只判别生成的selected和filler是否匹配
        dis_input = keras.layers.multiply([fake_profiles, keras.layers.add([selects_mask, fillers_mask])])
        output_validity = self.discriminator(dis_input)

        # custom loss
        def custom_generator_loss(input_template, output_fake, output_validity, ZR_mask, selects_mask):
            # loss_shilling
            loss_shilling = Lambda(lambda x: keras.backend.mean(
                (x * selects_mask - selects_mask * 5.) ** 2,
                axis=-1, keepdims=True))(output_fake)
            # loss_reconstruct
            loss_reconstruct = Lambda(lambda x: keras.backend.mean(
                ((x * selects_mask - selects_mask * input_template) * ZR_mask) ** 2,
                axis=-1, keepdims=True))(output_fake)
            # loss_adv
            loss_adv = Lambda(lambda x: keras.backend.binary_crossentropy(
                tf.ones_like(x), x))(output_validity)
            return keras.layers.add([loss_reconstruct, loss_shilling, loss_adv])

        ZR_mask = Input(shape=(self.n_items,))
        self.generator_train = Model(inputs=[real_profiles, fillers_mask, selects_mask, target_patch, ZR_mask],
                                     outputs=custom_generator_loss(real_profiles, fake_profiles, output_validity,
                                                                   ZR_mask, selects_mask))

        self.generator_train.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer_G)

    def build_generator(self):
        reg_rate = self.reg_rate_G
        ########################################
        # define keras model
        ########################################
        model = Sequential(name='generator')
        model.add(Dense(units=128, input_dim=self.n_items,
                        activation='sigmoid', use_bias=True,
                        # bias_regularizer=L2(reg_rate),
                        # kernel_regularizer=L2(reg_rate)
                        ))
        model.add(Dense(units=self.n_items,
                        activation='sigmoid', use_bias=True,
                        # bias_regularizer=L2(reg_rate),
                        # kernel_regularizer=L2(reg_rate)
                        ))
        # model.add(Dense(units=400, input_dim=self.n_items,
        #                 activation='sigmoid', use_bias=True,
        #                 bias_regularizer=L2(reg_rate),
        #                 kernel_regularizer=L2(reg_rate)
        #                 ))
        # model.add(Dense(133,
        #                 bias_regularizer=L2(reg_rate),
        #                 kernel_regularizer=L2(reg_rate)))
        # model.add(Dense(44,
        #                 bias_regularizer=L2(reg_rate),
        #                 kernel_regularizer=L2(reg_rate)))
        # model.add(Dense(14,
        #                 bias_regularizer=L2(reg_rate),
        #                 kernel_regularizer=L2(reg_rate)))
        # model.add(Dense(4,
        #                 bias_regularizer=L2(reg_rate),
        #                 kernel_regularizer=L2(reg_rate)))
        # model.add(Dense(units=len(self.selected_ids), activation='sigmoid',
        #                 bias_regularizer=L2(reg_rate),
        #                 kernel_regularizer=L2(reg_rate)))

        model.add(Lambda(lambda x: x * 5.0, name='gen_profiles'))

        model.summary()

        ########################################
        # build keras model
        ########################################
        # input
        real_profiles = Input(shape=(self.n_items,))
        fillers_mask = Input(shape=(self.n_items,))
        # template
        input_template = keras.layers.Multiply()([real_profiles, fillers_mask])  # real_profiles * fillers_mask
        # forward
        gen_output = model(input_template)
        ########################################
        # assemble fillers+selected+target to fake_profiles
        ########################################
        selects_mask = Input(shape=(self.n_items,))
        target_patch = Input(shape=(self.n_items,))
        selected_patch = keras.layers.Multiply()([gen_output, selects_mask])
        output_fake = keras.layers.add([input_template, selected_patch, target_patch])
        return Model([real_profiles, fillers_mask, selects_mask, target_patch], output_fake)

    def build_discriminator(self):
        reg_rate = self.reg_rate_D
        model = Sequential(name='discriminator')
        ########################################
        # input->hidden
        model.add(Dense(units=150, input_dim=self.n_items,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)
                        ))
        # stacked hidden layers
        model.add(Dense(150,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)))
        model.add(Dense(150,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)))
        # hidden -> output
        model.add(Dense(units=1,
                        activation='sigmoid', use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='random_uniform',
                        bias_regularizer=L2(reg_rate),
                        kernel_regularizer=L2(reg_rate)))
        ########################################
        model.summary()

        input_profile = Input(shape=(self.n_items,))
        validity = model(input_profile)
        return Model(input_profile, validity)

    def sample_fillers(self, real_profiles):
        fillers = np.zeros_like(real_profiles)
        filler_pool = set(range(self.n_items)) - set(self.selected_ids) - {self.target_id}
        # filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False)
        # sampled_cols = [filler_sampler([filler_pool, self.filler_num]) for _ in range(real_profiles.shape[0])]

        filler_sampler = lambda x: np.random.choice(size=self.filler_num, replace=False,
                                                    a=list(set(np.argwhere(x > 0).flatten()) & filler_pool))
        sampled_cols = [filler_sampler(x) for x in real_profiles]

        sampled_rows = np.repeat(np.arange(real_profiles.shape[0]), self.filler_num)
        fillers[sampled_rows, np.array(sampled_cols).flatten()] = 1
        return fillers

    def train(self):

        total_batch = math.ceil(len(self.template_idxs) / self.batch_size)
        idxs = np.random.permutation(self.template_idxs)  # shuffled ordering
        #

        d_loss_list, g_loss_list = [], []
        for i in range(total_batch):
            # ---------------------
            #  Prepare Input
            # ---------------------
            batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            # Adversarial ground truths
            valid_labels = np.ones_like(batch_set_idx)
            fake_labels = np.zeros_like(batch_set_idx)

            # Select a random batch of real_profiles
            real_profiles = self.train_data_array[batch_set_idx, :]
            # sample fillers
            fillers_mask = self.sample_fillers(real_profiles)
            # selected
            selects_mask = np.zeros_like(fillers_mask)
            selects_mask[:, self.selected_ids] = 1.
            # target
            target_patch = np.zeros_like(fillers_mask)
            target_patch[:, self.selected_ids] = 5.

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Generate a batch of new images
            fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
            # Train the discriminator

            d_loss_real = self.discriminator.train_on_batch(real_profiles * (fillers_mask + selects_mask), valid_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_profiles * (fillers_mask + selects_mask), fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            ZR_mask = (real_profiles == 0) * selects_mask

            pools = np.argwhere(ZR_mask)
            np.random.shuffle(pools)
            pools = pools[:math.floor(len(pools) * (1 - self.ZR_ratio))]
            ZR_mask[pools[:, 0], pools[:, 1]] = 0
            # Train the generator (to have the discriminator label samples as valid)
            # g_loss = self.combined.train_on_batch(real_profiles * fillers,valid)
            g_loss = self.generator_train.train_on_batch(
                [real_profiles, fillers_mask, selects_mask, target_patch, ZR_mask],
                valid_labels)

            d_loss_list.append(d_loss)
            g_loss_list.append(g_loss)
        return np.mean(d_loss_list), np.mean(g_loss_list)

    def execute(self):

        self.prepare_data()

        # Build and compile GAN Network
        self.build_network()

        if self.restore_model:
            self.restore(self.model_path)
            print("loading done.")

        else:
            for epoch in range(self.epochs):
                d_loss_cur, g_loss_cur = self.train()
                if self.verbose and epoch % self.T == 0:
                    print("epoch:%d\td_loss:%.4f\tg_loss:%.4f" % (epoch, d_loss_cur, g_loss_cur))

            self.save(self.model_path)
            print("training done.")

        metrics = self.test(victim='SVD', detect=True)
        print(metrics, flush=True)
        return

    def save(self, path):
        return

    def restore(self, path):
        return

    def generate_fakeMatrix(self):
        # Select a random batch of real_profiles
        idx = self.template_idxs[np.random.randint(0, len(self.template_idxs), self.attack_num)]
        real_profiles = self.train_data_array[idx, :]
        # sample fillers
        fillers_mask = self.sample_fillers(real_profiles)
        # selected
        selects_mask = np.zeros_like(fillers_mask)
        selects_mask[:, self.selected_ids] = 1.
        # target
        target_patch = np.zeros_like(fillers_mask)
        target_patch[:, self.target_id] = 5.

        # Generate
        fake_profiles = self.generator.predict([real_profiles, fillers_mask, selects_mask, target_patch])
        # selected patches
        selected_patches = fake_profiles[:, self.selected_ids]
        selected_patches = np.round(selected_patches)
        selected_patches[selected_patches > 5] = 5
        selected_patches[selected_patches < 1] = 1
        fake_profiles[:, self.selected_ids] = selected_patches

        return fake_profiles

    def generate_injectedFile(self, fake_array):
        super(AUSH, self).generate_injectedFile(fake_array)
