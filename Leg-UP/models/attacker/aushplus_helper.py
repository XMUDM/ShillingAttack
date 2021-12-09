# -*- coding: utf-8 -*-
# @Time       : 2020/12/3 20:03
# @Author     : chensi
# @File       : tkde.py
# @Software   : PyCharm
# @Desciption : None


import random
import numpy as np
import torch
from torch import nn

from utils.utils import *
from utils.loss import *
import higher

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
import time
import torch.nn.functional as F
import math
import torch.optim as optim


# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
class BaseGenerator(nn.Module):
    def __init__(self, device, input_dim):
        super(BaseGenerator, self).__init__()
        #
        self.input_dim = input_dim
        self.device = device

        """helper_tensor"""

        self.epsilon = torch.tensor(1e-4).to(self.device)  # 计算boundary
        self.helper_tensor = torch.tensor(2.5).to(device)
        pass

    def project(self, fake_tensor):
        fake_tensor.data = torch.round(fake_tensor)
        # fake_tensor.data = torch.where(fake_tensor < 1, torch.ones_like(fake_tensor).to(self.device), fake_tensor)
        fake_tensor.data = torch.where(fake_tensor < 0, torch.zeros_like(fake_tensor).to(self.device), fake_tensor)
        fake_tensor.data = torch.where(fake_tensor > 5, torch.tensor(5.).to(self.device), fake_tensor)
        #
        return fake_tensor

    def forward(self, input):
        raise NotImplementedError


class BaseDiscretGenerator(BaseGenerator):
    def __init__(self, device, input_dim):
        super(BaseDiscretGenerator, self).__init__(device, input_dim)

        self.min_boundary_value = torch.nn.Parameter(torch.rand([self.input_dim]), requires_grad=True)
        self.register_parameter("min_boundary_value", self.min_boundary_value)

        self.interval_lengths = torch.nn.Parameter(torch.rand([self.input_dim, 4]), requires_grad=True)
        self.register_parameter("interval_lengths", self.interval_lengths)

        pass

    def forward(self, input):
        # fake_tensor = (self.main(input) * self.helper_tensor) + self.helper_tensor
        # # project
        # fake_dsct_distribution, fake_dsct_value = self.project(fake_tensor)
        # return fake_dsct_value
        raise NotImplementedError

    def project(self, fake_tensor):

        Heaviside = HeaviTanh.apply

        boundary_values = self.get_boundary_values()

        cnt_ratings = fake_tensor.flatten()
        iids = np.expand_dims(np.arange(self.input_dim), 0).repeat(fake_tensor.shape[0], axis=0).flatten()
        boundary_values_per_rating = boundary_values[iids]

        def _project_helper(ratings, boundary_values_input):

            def get_target_dst_rating_prob(target_dst_rating, input_cnt_rating, boundary_values, device):
                # boundary_values = boundary_values.reshape([-1, 4])
                # input_cnt_rating = input_cnt_rating.reshape([-1])
                rating_prob = torch.ones(input_cnt_rating.shape[0]).to(self.device)
                for boundary_idx in range(5):
                    """
                    :param target_dst_rating: r_i_j
                    :param boundary_idx: k
                    :param input_cnt_rating: a_i_j
                    :param boundary_value: b_j_k
                    :return:
                    """
                    #
                    p_1 = torch.sign(target_dst_rating - boundary_idx - torch.tensor(0.5).to(device))
                    #
                    p_2 = input_cnt_rating - boundary_values[:, boundary_idx]
                    #
                    rating_prob *= Heaviside(p_1 * p_2, torch.tensor(1.).to(device))
                return rating_prob

            cur_dsct_distribution = []
            for rating_dsct in range(6):
                p = get_target_dst_rating_prob(rating_dsct, ratings, boundary_values_input, self.device)
                cur_dsct_distribution += [p]
            dsct_distribution = torch.cat([torch.unsqueeze(p, 1) for p in cur_dsct_distribution], 1)
            return dsct_distribution

        fake_dsct_distribution = _project_helper(cnt_ratings, boundary_values_per_rating).reshape(
            [-1, self.input_dim, 6])

        fake_dsct_value = torch.matmul(fake_dsct_distribution,
                                       torch.tensor(np.arange(0., 6.)).type(torch.float32).to(self.device))
        return fake_dsct_distribution, fake_dsct_value

    def project_old(self, fake_tensor):

        boundary_values = self.get_boundary_values()

        fake_dsct_distribution = []

        for iid in range(self.input_dim):

            cur_dsct_distribution = []
            for rating_dsct in range(6):
                rating_prob = torch.ones(fake_tensor.shape[0]).to(self.device)
                for boundary_idx in range(5):
                    rating_prob *= self.is_in_interval(rating_dsct,
                                                       boundary_idx,
                                                       fake_tensor[:, iid],
                                                       boundary_values[iid][boundary_idx])
                cur_dsct_distribution += [rating_prob]
            fake_dsct_distribution += [torch.cat([torch.unsqueeze(p, 1) for p in cur_dsct_distribution], 1)]
        fake_dsct_distribution = torch.cat([torch.unsqueeze(p, 1) for p in fake_dsct_distribution], 1)

        fake_dsct_value = torch.matmul(fake_dsct_distribution,
                                       torch.tensor(np.arange(6.0)).type(torch.float32).to(self.device))
        return fake_dsct_distribution, fake_dsct_value

    def get_boundary_values(self):

        boundary_values = torch.zeros([self.input_dim, 5]).to(self.device)
        boundary_values[:, 0] = self.min_boundary_value
        for i in range(1, 5):
            cur_interval_length = torch.relu(self.interval_lengths[:, i - 1]) + self.epsilon
            boundary_values[:, i] = boundary_values[:, i - 1] + cur_interval_length
        return boundary_values

    def is_in_interval(self, rating_dsct, boundary_idx, rating_cnt, boundary_value):
        tensor_aux_0_5 = torch.tensor(0.5).to(self.device)
        tensor_aux_1 = torch.tensor(1.).to(self.device)
        Heaviside = HeaviTanh.apply
        """

        :param rating_dsct: r_i_j
        :param boundary_idx: k
        :param rating_cnt: a_i_j
        :param boundary_value: b_j_k
        :return:
        """
        #
        p_1 = torch.sign(rating_dsct - boundary_idx - tensor_aux_0_5)
        #
        p_2 = rating_cnt - boundary_value
        #
        return Heaviside(p_1 * p_2, tensor_aux_1)


class RecsysGenerator(BaseGenerator):
    def __init__(self, device, init_tensor):
        super(RecsysGenerator, self).__init__(device, init_tensor.shape[1])
        """
        fake_parameter
        """
        fake_tensor = init_tensor.clone().detach().requires_grad_(True)
        self.fake_parameter = torch.nn.Parameter(fake_tensor, requires_grad=True)
        self.register_parameter("fake_tensor", self.fake_parameter)
        pass

    def forward(self, input=None):
        return None, self.project(self.fake_parameter * (input > 0))


class DiscretGenerator_AE(BaseDiscretGenerator):
    def __init__(self, device, p_dims, q_dims=None):
        super(DiscretGenerator_AE, self).__init__(device, input_dim=p_dims[0])
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.p_dims + self.q_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        # self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        # h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
            else:
                h = torch.nn.Tanh()(h)

        fake_tensor = (h * self.helper_tensor) + self.helper_tensor
        # project
        fake_dsct_distribution, fake_dsct_value = self.project(fake_tensor)
        sampled_filler = (input > 0)
        return fake_dsct_distribution, fake_dsct_value * sampled_filler

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class RoundGenerator_AE(BaseGenerator):
    def __init__(self, device, p_dims, q_dims=None):
        super(RoundGenerator_AE, self).__init__(device, input_dim=p_dims[0])
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.p_dims + self.q_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        # self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        # h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
            else:
                h = torch.nn.Tanh()(h)

        fake_tensor = (h * self.helper_tensor) + self.helper_tensor
        # project
        fake_dsct_value = self.project(fake_tensor)
        return None, fake_dsct_value * (input > 0)

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
# 最小值为1
class BaseGenerator_1(nn.Module):
    def __init__(self, device, input_dim):
        super(BaseGenerator_1, self).__init__()
        #
        self.input_dim = input_dim
        self.device = device

        """helper_tensor"""

        self.epsilon = torch.tensor(1e-4).to(self.device)  # 计算boundary
        self.helper_tensor = torch.tensor(2.5).to(device)
        pass

    def project(self, fake_tensor):
        fake_tensor.data = torch.round(fake_tensor)
        fake_tensor.data = torch.where(fake_tensor < 1, torch.ones_like(fake_tensor).to(self.device), fake_tensor)
        # fake_tensor.data = torch.where(fake_tensor < 0, torch.zeros_like(fake_tensor).to(self.device), fake_tensor)
        fake_tensor.data = torch.where(fake_tensor > 5, torch.tensor(5.).to(self.device), fake_tensor)
        #
        return fake_tensor

    def forward(self, input):
        raise NotImplementedError


class BaseDiscretGenerator_1(BaseGenerator):
    def __init__(self, device, input_dim):
        super(BaseDiscretGenerator_1, self).__init__(device, input_dim)

        # self.min_boundary_value = torch.nn.Parameter(torch.rand([self.input_dim]), requires_grad=True)
        self.min_boundary_value = torch.nn.Parameter(torch.ones([self.input_dim]), requires_grad=True)
        self.register_parameter("min_boundary_value", self.min_boundary_value)

        # self.interval_lengths = torch.nn.Parameter(torch.rand([self.input_dim, 3]), requires_grad=True)
        self.interval_lengths = torch.nn.Parameter(torch.ones([self.input_dim, 3]), requires_grad=True)
        self.register_parameter("interval_lengths", self.interval_lengths)

        pass

    def forward(self, input):
        # fake_tensor = (self.main(input) * self.helper_tensor) + self.helper_tensor
        # # project
        # fake_dsct_distribution, fake_dsct_value = self.project(fake_tensor)
        # return fake_dsct_value
        raise NotImplementedError

    def project_old(self, fake_tensor):

        boundary_values = self.get_boundary_values()


        fake_dsct_distribution = []

        for iid in range(self.input_dim):

            cur_dsct_distribution = []
            for rating_dsct in range(5):
                rating_prob = torch.ones(fake_tensor.shape[0]).to(self.device)
                for boundary_idx in range(4):
                    rating_prob *= self.is_in_interval(rating_dsct,
                                                       boundary_idx,
                                                       fake_tensor[:, iid],
                                                       boundary_values[iid][boundary_idx])
                cur_dsct_distribution += [rating_prob]
            fake_dsct_distribution += [torch.cat([torch.unsqueeze(p, 1) for p in cur_dsct_distribution], 1)]
        fake_dsct_distribution = torch.cat([torch.unsqueeze(p, 1) for p in fake_dsct_distribution], 1)

        fake_dsct_value = torch.matmul(fake_dsct_distribution,
                                       torch.tensor(np.arange(1., 6.)).type(torch.float32).to(self.device))
        return fake_dsct_distribution, fake_dsct_value

    def project(self, fake_tensor):

        Heaviside = HeaviTanh.apply

        boundary_values = self.get_boundary_values()

        cnt_ratings = fake_tensor.flatten()
        iids = np.expand_dims(np.arange(self.input_dim), 0).repeat(fake_tensor.shape[0], axis=0).flatten()
        boundary_values_per_rating = boundary_values[iids]

        def _project_helper(ratings, boundary_values_input):

            def get_target_dst_rating_prob(target_dst_rating, input_cnt_rating, boundary_values, device):
                # boundary_values = boundary_values.reshape([-1, 4])
                # input_cnt_rating = input_cnt_rating.reshape([-1])
                rating_prob = torch.ones(input_cnt_rating.shape[0]).to(self.device)
                for boundary_idx in range(4):
                    """
                    :param target_dst_rating: r_i_j
                    :param boundary_idx: k
                    :param input_cnt_rating: a_i_j
                    :param boundary_value: b_j_k
                    :return:
                    """
                    #
                    p_1 = torch.sign(target_dst_rating - boundary_idx - torch.tensor(0.5).to(device))
                    #
                    p_2 = input_cnt_rating - boundary_values[:, boundary_idx]
                    #
                    rating_prob *= Heaviside(p_1 * p_2, torch.tensor(1.).to(device))
                return rating_prob

            cur_dsct_distribution = []
            for rating_dsct in range(5):
                p = get_target_dst_rating_prob(rating_dsct, ratings, boundary_values_input, self.device)
                cur_dsct_distribution += [p]
            dsct_distribution = torch.cat([torch.unsqueeze(p, 1) for p in cur_dsct_distribution], 1)
            return dsct_distribution

        fake_dsct_distribution = _project_helper(cnt_ratings, boundary_values_per_rating).reshape(
            [-1, self.input_dim, 5])

        fake_dsct_value = torch.matmul(fake_dsct_distribution,
                                       torch.tensor(np.arange(1., 6.)).type(torch.float32).to(self.device))
        return fake_dsct_distribution, fake_dsct_value

    def get_boundary_values(self):

        boundary_values = torch.zeros([self.input_dim, 4]).to(self.device)
        boundary_values[:, 0] = self.min_boundary_value
        for i in range(1, 4):
            cur_interval_length = torch.relu(self.interval_lengths[:, i - 1]) + self.epsilon
            boundary_values[:, i] = boundary_values[:, i - 1] + cur_interval_length
        return boundary_values

    def is_in_interval(self, rating_dsct, boundary_idx, rating_cnt, boundary_value):
        tensor_aux_0_5 = torch.tensor(0.5).to(self.device)
        tensor_aux_1 = torch.tensor(1.).to(self.device)
        Heaviside = HeaviTanh.apply
        """
    
        :param rating_dsct: r_i_j
        :param boundary_idx: k
        :param rating_cnt: a_i_j
        :param boundary_value: b_j_k
        :return:
        """
        #
        p_1 = torch.sign(rating_dsct - boundary_idx - tensor_aux_0_5)
        #
        p_2 = rating_cnt - boundary_value
        #
        return Heaviside(p_1 * p_2, tensor_aux_1)


class DiscretGenerator_AE_1(BaseDiscretGenerator_1):
    def __init__(self, device, p_dims, q_dims=None):
        super(DiscretGenerator_AE_1, self).__init__(device, input_dim=p_dims[0])
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.p_dims + self.q_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        # self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        # h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
            else:
                h = torch.nn.Tanh()(h)

        fake_tensor = (h * self.helper_tensor) + self.helper_tensor
        # project

        fake_dsct_distribution, fake_dsct_value = self.project(fake_tensor)

        sampled_filler = (input > 0)

        # sampled_filler = (torch.rand(fake_dsct_value.shape) < (90 / 1924)).float()

        # filler_num = np.sum(sampled_filler.detach().cpu().numpy()*(fake_dsct_value.detach().cpu().numpy()>0),1).mean()
        # if filler_num<90:
        return fake_dsct_distribution, fake_dsct_value * sampled_filler

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class RoundGenerator_AE_1(BaseGenerator_1):
    def __init__(self, device, p_dims, q_dims=None):
        super(RoundGenerator_AE_1, self).__init__(device, input_dim=p_dims[0])
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.p_dims + self.q_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                     d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        # self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        # h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
            else:
                h = torch.nn.Tanh()(h)

        fake_tensor = (h * self.helper_tensor) + self.helper_tensor
        # project
        fake_dsct_value = self.project(fake_tensor)

        sampled_filler = (input > 0)
        return None, fake_dsct_value * sampled_filler

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class DiscretRecsysGenerator_1(BaseDiscretGenerator_1):
    def __init__(self, device, init_tensor):
        super(DiscretRecsysGenerator_1, self).__init__(device, init_tensor.shape[1])
        """
        fake_parameter
        """
        fake_tensor = init_tensor.clone().detach().requires_grad_(True)
        self.fake_parameter = torch.nn.Parameter(fake_tensor, requires_grad=True)
        self.register_parameter("fake_tensor", self.fake_parameter)
        pass

    def forward(self, input=None):
        fake_dsct_distribution, fake_dsct_value = self.project(self.fake_parameter)
        sampled_filler = (input > 0)
        return fake_dsct_distribution, fake_dsct_value * sampled_filler


# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================


class HeaviTanh(torch.autograd.Function):
    """
    Approximation of the heaviside step function as
    h(x,k) = \frac{1}{2} + \frac{1}{2} \text{tanh}(k x)
    """

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x, k)

        def heaviside(data):
            """
            A `heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`_
            that truncates numbers <= 0 to 0 and everything else to 1.
            .. math::
                H[n]=\\begin{cases} 0, & n <= 0, \\ 1, & n \g 0, \end{cases}
            """
            return torch.where(
                data <= torch.zeros_like(data), torch.zeros_like(data), torch.ones_like(data),
            )

        return heaviside(x)  # 0.5 + 0.5 * torch.tanh(k * x)

    @staticmethod
    def backward(ctx, dy):
        x, k, = ctx.saved_tensors
        dtanh = 1 - (x * k).tanh().pow(2)
        return dy * dtanh, None


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):

        return self.main(input)



class BaseTrainer(object):
    def __init__(self):
        self.args = None

        self.n_users = None
        self.n_items = None

        self.net = None
        self.optimizer = None
        self.metrics = None
        self.golden_metric = "Recall@50"

    @staticmethod
    def minibatch(*tensors, **kwargs):
        """Mini-batch generator for pytorch tensor."""
        batch_size = kwargs.get('batch_size', 128)

        if len(tensors) == 1:
            tensor = tensors[0]
            for i in range(0, len(tensor), batch_size):
                yield tensor[i:i + batch_size]
        else:
            for i in range(0, len(tensors[0]), batch_size):
                yield tuple(x[i:i + batch_size] for x in tensors)

    @staticmethod
    def mult_ce_loss(data, logits):

        # ========================================
        # surrogate network loss function
        # ========================================
        # Func_WeightedMSELoss = lambda weight, input, target: \
        #     torch.where(target > 2, torch.tensor(weight), torch.tensor(1.)) \
        #     * MSELoss(reduce=False, size_average=False)(input, target)
        #
        # Func_MSELoss = MSELoss(reduce=False, size_average=False)

        # adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
        # # Copy fmodel's parameters to default trainer.net().
        # model.load_state_dict(fmodel.state_dict())
        """Multi-class cross-entropy loss."""
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs * data

        instance_data = data.sum(1)
        instance_loss = loss.sum(1)
        # Avoid divide by zeros.
        res = instance_loss / (instance_data + 0.1)  # PSILON)
        return res

    @staticmethod
    def weighted_mse_loss(data, logits, weight_pos=1, weight_neg=0):
        """Mean square error loss."""
        weights = torch.ones_like(data) * weight_neg
        weights[data > 0] = weight_pos
        res = weights * (data - logits) ** 2
        return res.sum(1)

    @staticmethod
    def _array2sparsediag(x):
        values = x
        indices = np.vstack([np.arange(x.size), np.arange(x.size)])

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = [x.size, x.size]

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    @property
    def _initialized(self):
        return self.net is not None

    def _initialize(self):
        """Initialize model and optimizer."""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        """Generate a top-k recommendation (ranked) list."""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def train_epoch(self, data):
        """Train model for one epoch"""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def train_epoch_wrapper(self, train_data, epoch_num):
        """Wrapper for train_epoch with some logs."""
        time_st = time.time()
        epoch_loss = self.train_epoch(train_data)
        print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
            time.time() - time_st, epoch_num, epoch_loss))

    def evaluate_epoch(self, train_data, test_data, epoch_num):
        """Evaluate model performance on test data."""
        t1 = time.time()

        n_rows = train_data.shape[0]
        n_evaluate_users = test_data.shape[0]

        total_metrics_len = sum(len(x) for x in self.metrics)
        total_val_metrics = np.zeros([n_rows, total_metrics_len], dtype=np.float32)

        recommendations = self.recommend(train_data, top_k=100)

        valid_rows = list()
        for i in range(train_data.shape[0]):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = test_data[i].indices
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            metric_results = list()
            for metric in self.metrics:
                result = metric(targets, recs)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)

        # Average evaluation results by user.
        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()

        # Summary evaluation results into a dict.
        # ind, result = 0, OrderedDict()
        # for metric in self.metrics:
        #     values = avg_val_metrics[ind:ind + len(metric)]
        #     if len(values) <= 1:
        #         result[str(metric)] = values[0]
        #     else:
        #         for name, value in zip(str(metric).split(','), values):
        #             result[name] = value
        #     ind += len(metric)
        #
        # print("Evaluation [{:.1f} s],  epoch: {}, {} ".format(
        #     time.time() - t1, epoch_num, str(result)))
        # return result

    def fit(self, train_data, test_data):
        """Full model training loop."""
        if not self._initialized:
            self._initialize()

        if self.args.save_feq > self.args.epochs:
            raise ValueError("Model save frequency should be smaller than"
                             " total training epochs.")

        start_epoch = 1
        best_checkpoint_path = ""
        best_perf = 0.0
        for epoch_num in range(start_epoch, self.args.epochs + 1):
            # Train the model.
            self.train_epoch_wrapper(train_data, epoch_num)
            if epoch_num % self.args.save_feq == 0:
                result = self.evaluate_epoch(train_data, test_data, epoch_num)
                # Save model checkpoint if it has better performance.
                # if result[self.golden_metric] > best_perf:
                #     str_metric = "{}={:.4f}".format(self.golden_metric,
                #                                     result[self.golden_metric])
                #     print("Having better model checkpoint with"
                #           " performance {}".format(str_metric))
                #     checkpoint_path = os.path.join(
                #         self.args.output_dir,
                #         self.args.model['model_name'])
                #     save_checkpoint(self.net, self.optimizer,
                #                     checkpoint_path,
                #                     epoch=epoch_num)
                #
                #     best_perf = result[self.golden_metric]
                #     best_checkpoint_path = checkpoint_path

        # Load best model and evaluate on test data.
        print("Loading best model checkpoint.")
        self.restore(best_checkpoint_path)
        self.evaluate_epoch(train_data, test_data, -1)
        return

    def restore(self, path):
        return


class WeightedMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim):
        super(WeightedMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dim

        self.Q = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        self.P = nn.Parameter(
            torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.1))
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())


class WMFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, device, hidden_dim, lr, weight_decay, batch_size,
                 weight_pos, weight_neg, verbose=False):
        super(WMFTrainer, self).__init__()
        self.device = device
        #
        self.n_users = n_users
        self.n_items = n_items
        #
        self.hidden_dim = hidden_dim
        #
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        #
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        #
        self.verbose = verbose

        pass

    def _initialize(self):
        self.net = WeightedMF(n_users=self.n_users,
                              n_items=self.n_items,
                              hidden_dim=self.hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        self.dim = self.net.dim

    def fit_adv(self, data_tensor, epoch_num, unroll_steps):
        self._initialize()

        import higher
        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")
        #
        data_tensor = data_tensor.to(self.device)
        n_rows = data_tensor.shape[0]
        idx_list = np.arange(n_rows)
        #
        model = self.net.to(self.device)
        #
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0

            for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
                # Compute loss

                # TODO detach()
                loss = self.weighted_mse_loss(data=data_tensor[batch_idx].detach(),
                                              logits=model(user_id=batch_idx),
                                              weight_pos=self.weight_pos,
                                              weight_neg=self.weight_neg).sum()
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.verbose:
                print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                    time.time() - t1, i, epoch_loss), flush=True)

        with higher.innerloop_ctx(model, self.optimizer) as (fmodel, diffopt):
            if self.verbose:
                print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
                    # Compute loss
                    # ===========warning=================

                    loss = self.weighted_mse_loss(data=data_tensor[batch_idx],
                                                  logits=fmodel(user_id=batch_idx),
                                                  weight_pos=self.weight_pos,
                                                  weight_neg=self.weight_neg).sum()
                    # ====================================
                    epoch_loss += loss.item()
                    diffopt.step(loss)
                if self.verbose:
                    print("Training (higher mode) [{:.1f} s],"
                          " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss), flush=True)
            #
            if self.verbose:
                print("Finished surrogate model training,"
                      " {} copies of surrogate model params.".format(len(fmodel._fast_params)), flush=True)

            fmodel.eval()
            predictions = fmodel()
        return predictions  # adv_loss  # .item(), adv_grads[-n_fakes:, ]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in self.minibatch(
                    idx_list, batch_size=self.args.valid_batch_size):
                batch_data = data[batch_idx].toarray()

                preds = model(user_id=batch_idx)
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[batch_data.nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations


class ItemAE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(ItemAE, self).__init__()
        self.q_dims = [input_dim] + [hidden_dims]
        self.p_dims = self.q_dims[::-1]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.q_dims[:-1], self.q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.p_dims[:-1], self.p_dims[1:])])

    def encode(self, input):
        h = input
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            h = torch.tanh(h)
        return h

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, input):
        z = self.encode(input)
        return self.decode(z)

    def loss(self, data, outputs):
        return BaseTrainer.weighted_mse_loss(data=data, logits=outputs)


class ItemAETrainer(BaseTrainer):
    def __init__(self, n_users, n_items, hidden_dims, device, lr, l2, batch_size, weight_pos, weight_neg,
                 verbose=False):
        super(ItemAETrainer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dims = hidden_dims

        self.device = device
        self.lr = lr
        self.l2 = l2
        self.batch_size = batch_size
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.device = device
        self.verbose = verbose

        pass

    def _initialize(self):
        self.net = ItemAE(self.n_users, self.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr, weight_decay=self.l2)

    def train_epoch(self, data):
        # Transpose the data first for ItemVAE.
        data = data.transpose()

        n_rows = data.shape[0]
        n_cols = data.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        model.train()
        np.random.shuffle(idx_list)

        epoch_loss = 0.0
        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))

        for batch_idx in self.minibatch(idx_list, batch_size=batch_size):
            batch_tensor = data[batch_idx].to(self.device)
            # Compute loss
            outputs = model(batch_tensor)
            loss = model.loss(data=batch_tensor,
                              outputs=outputs).sum()
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss

    def fit_adv(self, data_tensor, epoch_num, unroll_steps, ):
        import higher

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        self._initialize()

        data_tensor = data_tensor.to(self.device)
        # target_tensor = torch.zeros_like(data_tensor)
        # target_tensor[:, target_items] = 1.0
        data_tensor = data_tensor.t()

        n_rows = data_tensor.shape[0]
        n_cols = data_tensor.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.batch_size if self.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in self.minibatch(idx_list, batch_size=batch_size):
                # TODO detach()
                batch_tensor = data_tensor[batch_idx].detach()
                # Compute loss
                outputs = model(batch_tensor)
                loss = model.loss(data=batch_tensor, outputs=outputs).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self.verbose:  # and i%20==0:
                print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                    time.time() - t1, i, epoch_loss))

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            if self.verbose:
                print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                epoch_loss = 0.0
                fmodel.train()
                for batch_idx in self.minibatch(idx_list, batch_size=batch_size):
                    batch_tensor = data_tensor[batch_idx]
                    # Compute loss
                    outputs = fmodel(batch_tensor)
                    loss = fmodel.loss(data=batch_tensor, outputs=outputs).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)
                if self.verbose:
                    print("Training (higher mode) [{:.1f} s],"
                          " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss))
            if self.verbose:
                print("Finished surrogate model training,"
                      " {} copies of surrogate model params.".format(len(fmodel._fast_params)))

            fmodel.eval()
            all_preds = list()
            for batch_idx in self.minibatch(np.arange(n_rows),
                                            batch_size=batch_size):
                all_preds += [fmodel(data_tensor[batch_idx])]
            predictions = torch.cat(all_preds, dim=0).t()

            # # Compute adversarial (outer) loss.
            # adv_loss = self.mult_ce_loss(
            #     logits=predictions[:-n_fakes, ],
            #     data=target_tensor[:-n_fakes, ]).sum()
            # adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            # # Copy fmodel's parameters to default trainer.net().
            # model.load_state_dict(fmodel.state_dict())

        return predictions  # adv_loss.item(), adv_grads.t()[-n_fakes:, :]



class SVDpp(nn.Module):
    def __init__(self, n_users, n_items, hidden_dims, data):
        super(SVDpp, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dims[0]
        self.data = data
        self.Q1 = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        self.Q2 = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        self.P = nn.Parameter(
            torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.1))
        self.bu = nn.Parameter(torch.zeros(self.n_users))
        self.bi = nn.Parameter(torch.zeros(self.n_items))
        # store each users' interaction history
        self.Ni = list()
        for user in self.data:
            self.Ni.append(user.nonzero().squeeze(1))
        self.u = self.data.float().mean()

    def forward(self, user_id=None, item_id=None):
        # bias computing
        bu = self.bu.expand((self.n_items, self.n_users)).t()
        bi = self.bi.expand((self.n_users, self.n_items))
        b = bu + bi
        # user features computing
        P = list()
        for i in self.Ni:
            yi = self.Q2[i]
            Yi = self.Q2[i].sum(dim=0)
            length = len(yi)
            P.append(Yi / math.sqrt(length))
        P = torch.cat(P).view((self.n_users, self.dim))
        P = P + self.P
        if user_id is None and item_id is None:
            return torch.sigmoid(torch.mm(P, self.Q1.t()) + b[[user_id]] + self.u) * 5
        if user_id is not None:
            return torch.sigmoid(torch.mm(P[[user_id]], self.Q1.t()) + b[[user_id]] + self.u) * 5
        if item_id is not None:
            return torch.sigmoid(torch.mm(P, self.Q1[[item_id]].t()) + b[[user_id]] + self.u) * 5


class SVDppTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, hidden_dims, device, lr, l2, batch_size, weight_alpha):
        super(SVDppTrainer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dims = hidden_dims
        self.device = device
        self.lr = lr
        self.l2 = l2
        self.batch_size = batch_size
        self.weight_alpha = weight_alpha

    def _initialize(self, data):
        self.net = SVDpp(
            n_users=self.n_users,
            n_items=self.n_items,
            hidden_dims=self.hidden_dims,
            data=data
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.l2)

    def fit_adv(self, data_tensor, epoch_num, unroll_steps):
        self._initialize(data_tensor)

        import higher

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        data_tensor = data_tensor.to(self.device)
        n_rows = data_tensor.shape[0]
        n_cols = data_tensor.shape[1]
        idx_list = np.arange(n_rows)

        model = self.net.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.batch_size
                      if self.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(
                    idx_list, batch_size=batch_size):
                # Compute loss
                loss = mse_loss(data=data_tensor[batch_idx],
                                logits=model(user_id=batch_idx),
                                weight=self.weight_alpha).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                time.time() - t1, i, epoch_loss))

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in minibatch(
                        idx_list, batch_size=batch_size):
                    # Compute loss
                    loss = mse_loss(data=data_tensor[batch_idx],
                                    logits=fmodel(user_id=batch_idx),
                                    weight=self.weight_alpha).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)

                print("Training (higher mode) [{:.1f} s],"
                      " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss))

            print("Finished surrogate model training,"
                  " {} copies of surrogate model params.".format(len(fmodel._fast_params)))

            fmodel.eval()
            predictions = fmodel()
        return predictions.squeeze(0)


class NMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim, data):
        super(NMF, self).__init__()
        self.n_users, self.n_items = n_users, n_items
        self.hideen_dim = hidden_dim
        self.data = data

        self.scale = torch.sqrt(torch.mean(self.data.detach()) / self.hideen_dim)
        W = torch.abs(torch.rand([self.n_users, self.hideen_dim]) * self.scale)
        H = torch.abs(torch.rand([self.hideen_dim, self.n_items]) * self.scale)
        self.W = torch.nn.Parameter(W, requires_grad=True)
        self.H = torch.nn.Parameter(H, requires_grad=True)

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.W, self.H)
        if user_id is not None:
            return torch.mm(self.W[[user_id]], self.H)
        if item_id is not None:
            return torch.mm(self.W, self.H[[item_id]])



class NMFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items,
                 batch_size, device,
                 k=128, solver='autograd', eps=1e-7,
                 alpha=0.99,
                 loss='l2',
                 lr=1e-2):
        super(NMFTrainer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size
        self.k = k
        self.loss = loss
        self.lr = lr
        self.alpha = alpha
        self.solver = solver
        self.eps = eps
        self.device = device

    @staticmethod
    def weighted_mse_loss(data, logits, weight_pos=1.0, weight_neg=0.0):
        """Mean square error loss."""
        weights = torch.ones_like(data) * weight_neg
        weights[data > 0] = weight_pos
        res = weights * (data - logits) ** 2
        return res.sum(1)

    @staticmethod
    def l2(x, y):
        return torch.nn.MSELoss()(x, y)

    @staticmethod
    def kl_dev(x, y):
        return (x * torch.log(x / y) - x + y).mean()

    def _initialize(self, data_tensor):
        self.net = NMF(self.n_users, self.n_items, self.k, data_tensor)
        # for autograd solver
        self.opt = torch.optim.RMSprop(self.net.parameters(), alpha=self.alpha, lr=self.lr, weight_decay=1e-6)

    def plus(self, X):
        X[X < 0] = self.eps
        return X

    def fit_adv(self, data_tensor, epoch_num, unroll_steps):
        self._initialize(data_tensor)

        import higher

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        data_tensor = data_tensor.to(self.device)
        n_rows = data_tensor.shape[0]
        n_cols = data_tensor.shape[1]
        idx_list = np.arange(n_rows)

        model = self.net.to(self.device)
        optimizer = self.opt

        batch_size = (self.batch_size
                      if self.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(
                    idx_list, batch_size=self.batch_size):
                # Compute loss
                # loss = NMFTrainer.weighted_mse_loss(data=data_tensor[batch_idx],
                #                                     logits=model(user_id=batch_idx),
                #                                     weight_pos=1.0,
                #                                     weight_neg=-1.0 * self.eps).sum()
                loss = NMFTrainer.l2(data_tensor[batch_idx], model(user_id=batch_idx))
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                for p in model.parameters():
                    p.data = self.plus(p.data)

            print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                time.time() - t1, i, epoch_loss))

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in minibatch(
                        idx_list, batch_size=self.batch_size):
                    # Compute loss
                    # loss = NMFTrainer.weighted_mse_loss(data=data_tensor[batch_idx],
                    #                                     logits=fmodel(user_id=batch_idx),
                    #                                     weight_pos=1.0,
                    #                                     weight_neg=-1.0*self.eps).sum()
                    loss = NMFTrainer.l2(data_tensor[batch_idx], fmodel(user_id=batch_idx))
                    epoch_loss += loss.item()
                    diffopt.step(loss)
                    for p in fmodel.parameters():
                        p.data = self.plus(p.data)

                print("Training (higher mode) [{:.1f} s],"
                      " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss))

            print("Finished surrogate model training,"
                  " {} copies of surrogate model params.".format(len(fmodel._fast_params)))

            fmodel.eval()
            predictions = fmodel()
        return predictions


from numpy.random import RandomState


class PMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=128, is_sparse=False, no_cuda=None):
        super(PMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.no_cuda = no_cuda
        self.random_state = RandomState(1)

        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=is_sparse)
        self.user_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()

        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=is_sparse)
        self.item_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()

        self.ub = nn.Embedding(n_users, 1)
        self.ib = nn.Embedding(n_items, 1)
        self.ub.weight.data.uniform_(-.01, .01)
        self.ib.weight.data.uniform_(-.01, .01)

    def forward(self, user_id, item_id):
        user_h1 = self.user_embeddings(user_id)
        item_h1 = self.item_embeddings(item_id).T
        R_h = torch.mm(user_h1, item_h1) + self.ub(user_id) + self.ib(item_id).T
        return R_h


class PMFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, device, hidden_dim, lr, weight_decay, batch_size,
                 momentum, verbose=False):
        super(PMFTrainer, self).__init__()
        self.device = device
        #
        self.n_users = n_users
        self.n_items = n_items
        #
        self.hidden_dim = hidden_dim
        #
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        #
        self.verbose = verbose

        pass

    def _initialize(self):
        self.net = PMF(n_users=self.n_users,
                       n_items=self.n_items,
                       n_factors=self.hidden_dim).to(self.device)

        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        for name, param in self.net.named_parameters():
            print(name)

    def fit_adv(self, data_tensor, epoch_num, unroll_steps):
        self._initialize()

        import higher
        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")
        #
        data_tensor = data_tensor.to(self.device)
        n_rows = data_tensor.shape[0]
        idx_list = np.arange(n_rows)
        #
        model = self.net.to(self.device)
        #
        user_idx = np.array(range(self.n_users), dtype=np.int16)
        item_idx = np.array(range(self.n_items), dtype=np.int16)
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0

            for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
                # Compute loss

                loss = mse_loss(data_tensor[batch_idx].float(),
                                model(user_id=torch.tensor(batch_idx).long(),
                                item_id=torch.tensor(item_idx).long()).float(),
                                1).sum()
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            if self.verbose:
                print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                    time.time() - t1, i, epoch_loss), flush=True)

        with higher.innerloop_ctx(model, self.optimizer) as (fmodel, diffopt):
            if self.verbose:
                print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in self.minibatch(idx_list, batch_size=self.batch_size):
                    # Compute loss
                    # ===========warning=================

                    loss = mse_loss(data_tensor[batch_idx].float(),
                                fmodel(user_id=torch.tensor(batch_idx).long(),
                                item_id=torch.tensor(item_idx).long()).float(),
                                1).sum()
                    # ====================================
                    epoch_loss += loss.item()
                    diffopt.step(loss)
                if self.verbose:
                    print("Training (higher mode) [{:.1f} s],"
                          " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss), flush=True)
            #
            if self.verbose:
                print("Finished surrogate model training,"
                      " {} copies of surrogate model params.".format(len(fmodel._fast_params)), flush=True)

            fmodel.eval()
            predictions = fmodel(torch.tensor(user_idx).long(), torch.tensor(item_idx).long())
            print(predictions)
        return predictions

