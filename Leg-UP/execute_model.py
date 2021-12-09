# -*- coding: utf-8 -*-
# @Time       : 2020/11/29 11:59
# @Author     : chensi
# @File       : execute_model.py
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

from importlib import import_module
import sys


model2lib_dict = {
    # attacker
    'RandomAttacker': 'models.attacker.attacker',
    'AverageAttacker': 'models.attacker.attacker',
    'BandwagonAttacker': 'models.attacker.attacker',
    'SegmentAttacker': 'models.attacker.attacker',
    #
    'WGANAttacker': 'models.attacker.attacker',
    'DCGANAttacker': 'models.attacker.attacker',
    #
    'AUSH': 'models.attacker.aush',
    #
    'AUSHplus': 'models.attacker.aushplus',
    'AIA': 'models.attacker.aushplus',
    'AUSHplus_SR': 'models.attacker.aushplus',
    'AUSHplus_woD': 'models.attacker.aushplus',
    'AUSHplus_SF': 'models.attacker.aushplus',
    'AUSHplus_inseg': 'models.attacker.aushplus',
}


def execute_model(model_type, model_name):

    try:
        try:
            model_lib_str = 'models.%s.%s' % (model_type.lower(),
                                              model_type[0].upper() + model_type[1:].lower())
            model_lib = import_module(model_lib_str)
            model = getattr(model_lib, model_name)()
        except:
            model_lib_str = 'utils.%s' % (model_type.lower())
            model_lib = import_module(model_lib_str)
            model = getattr(model_lib, model_name)()
    except:
        # try:
            model_lib_str = model2lib_dict[model_name]
            model_lib = import_module(model_lib_str)
            model = getattr(model_lib, model_name)()
        # except:
        #     print('Not found:', model_type, model_name)
        #     exit()

    model.execute()
    print('success.')


model_lib = sys.argv[sys.argv.index('--exe_model_lib') + 1]
model_name = sys.argv[sys.argv.index('--exe_model_class') + 1]
execute_model(model_lib, model_name)
