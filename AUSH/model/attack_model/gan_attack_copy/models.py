# -*- coding: utf-8 -*-
# @Time       : 2020/9/18 13:52
# @Author     : chensi
# @File       : models.py
# @Software   : PyCharm
# @Desciption : None

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import math


# import math
class CopyGanAttacker:
    def __init__(self, dataset_class, target_id, filler_num, attack_num, filler_method):
        # data set info
        self.dataset_class = dataset_class
        self.num_user = dataset_class.n_users
        self.num_item = dataset_class.n_items
        self.rating_matrix = dataset_class.train_matrix.toarray()  # tf.constant()

        # attack info
        self.target_id = target_id
        self.filler_num = filler_num
        self.attack_num = attack_num
        self.filler_method = filler_method

    def build_model(self):
        # define place_holder
        # self.user_vector = tf.placeholder(tf.int32, [None, self.num_item])
        # self.item_vector = tf.placeholder(tf.int32, [None, self.num_item])
        self.sampled_template = tf.placeholder(tf.int32, [self.args.batch_size, self.num_item])
        self.batch_filler_index = tf.placeholder(tf.int32, [None, self.args.batch_size])
        # user/item embedding
        # c = tf.constant(c)
        user_embedding = self.towerMlp(self.rating_matrix, self.num_item, self.args.embedding_dim)
        item_embedding = self.towerMlp(self.rating_matrix.transpose(), self.num_user, self.args.embedding_dim)

        """
        copy net  
        p_copy(j)=sigmoid (w x j’s item embedding + w x u’s user embedding + b)"""
        with tf.name_scope("copyNet"):
            w1 = tf.get_variable('w1', [self.args.embedding_dim, self.num_item])
            p1 = tf.matmul(tf.nn.embedding_lookup(user_embedding, self.batch_filler_index), w1)  # batch*item_num
            w2 = tf.get_variable('w2', [self.args.embedding_dim, 1])
            p2 = tf.matmul(item_embedding, w2)  # item_num*1
            b = tf.get_variable('b', [self.item_num])
            copy_prob = tf.nn.sigmoid(p1 + p2 + b)  # batch*item_num
        """
        generate net
        p_gen(j=r)
        """
        with tf.name_scope("genNet"):
            gen_probabilitiy_list = []
            for i in range(5):
                with tf.name_scope("s_%d" % i):
                    w1 = tf.get_variable('w1', [self.args.embedding_dim, self.num_item])
                    p1 = tf.matmul(tf.nn.embedding_lookup(user_embedding, self.batch_filler_index),
                                   w1)  # batch*item_num
                    w2 = tf.get_variable('w2', [self.args.embedding_dim, 1])
                    p2 = tf.matmul(item_embedding, w2)  # item_num*1
                    b = tf.get_variable('b', [self.item_num])
                    gen_probability = p1 + p2 + b
                    gen_probabilitiy_list.append(tf.expand_dims(gen_probability, 2))  # batch*item_num*1
            gen_rating_distri = tf.nn.softmax(tf.concat(gen_probabilitiy_list, axis=2))  # batch*item_num*5
        """
        Rating
        rating p(r) = p_copy(j) x p_copy(j=r) + (1-p_copy(j)) x p_gen(j=r)
        """
        copy_rating_distri = tf.reshape(tf.expand_dims(tf.one_hot(self.sampled_template, 5), 3),
                                        [self.args.batch_size, -1, 5])
        rating_distri = copy_prob * copy_rating_distri + (1 - copy_prob) * gen_rating_distri  # batch*item_num*5
        rating_value = tf.tile(tf.constant([[[1., 2., 3., 4., 5.]]]), [self.args.batch_size, self.num_item, 1])
        fake_profiles = tf.reduce_sum(rating_distri * rating_value, 2)

        """
        loss function
        """
        with tf.name_scope("Discriminator"):
            D_real = self.towerMlp(self.sampled_template, self.num_item, 1)
            D_fake = self.towerMlp(fake_profiles, self.num_item, 1)

        """
        loss function
        """
        with tf.name_scope("loss_D"):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)),
                name="loss_real")
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)),
                name="loss_fake")
            loss_D = d_loss_real + d_loss_fake
        with tf.name_scope("loss_G"):
            # reconstruction loss
            loss_rec = tf.reduce_mean(tf.square(fake_profiles - self.sampled_template))
            # adversial loss
            loss_adv = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
            loss_G = loss_rec + loss_adv

    def towerMlp(self, input, inputDim, outputDim):
        dim, x = inputDim // 2, input
        while dim > outputDim:
            layer = tf.layers.dense(
                inputs=x,
                units=dim,
                kernel_initializer=tf.random_normal_initializer,
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
            dim, x = dim // 2, layer
        output = tf.layers.dense(
            inputs=x,
            units=outputDim,
            kernel_initializer=tf.random_normal_initializer,
            activation=tf.nn.sigmoid,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        return output
