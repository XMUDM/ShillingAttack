# -*- coding: utf-8 -*-
# @Time       : 2019/8/30 9:24
# @Author     : chensi
# @File       : plot.py
# @Software   : PyCharm
# @Desciption : None

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

attack_methods = ["segment", "average", "random", "bandwagon", "gan"]
attack_name = ["Segment", "Random", "Average", "Bandwagon", "Ours"]
attack_method = "segment"
# random = [155, 383, 920, 941, 892]
# tail = [1480, 844, 1202, 1301, 2035]
# targets = random + tail
random = [5, 395, 181, 565, 254]
tail = [601, 623, 619, 64, 558]
targets = random + tail
# targets = [62, 1077, 785, 1419, 1257] + [1319, 1612, 1509, 1545, 1373]
# for attack_method in attack_methods:
#     # dir = '../results/ciao_DegreeSAD/' + attack_method
#     dir = '../results/filmTrust_0903_FAP/' + attack_method
#     pathDir = os.listdir(dir)
#     data_to_write = []
#     iid_idx = 0
#     for i in range(len(pathDir)):
#         # if "5-fold-cv" not in pathDir[i]: continue
#         iid = targets[iid_idx]
#         iid_idx += 1
#         # load result
#         lines = []
#         if 'FAP' not in pathDir[i]: continue
#         with open(dir + '/' + pathDir[i], 'r') as fin:
#             for line in fin:
#                 lines.append(line)
#         res = lines[3].strip('\n').split(' ')
#         while '' in res: res.remove('')
#         res = [str(iid)] + res
#         data_to_write.append('\t'.join(res))
#     with open(dir + '/' + "result_" + attack_method, 'w') as fout:
#         fout.write('\n'.join(data_to_write))

names = ['iid', 'label', 'precision', 'recall', 'f1', 'support']
# pre_results = {}
# recall_results = {}
P, R, N = [], [], []
for i in range(len(attack_methods)):
    attack_method = attack_methods[i]
    path = '../results/filmTrust_0903_FAP/' + attack_method + "/result_" + attack_method
    # path = '../results/ml100k_DegreeSAD/' + attack_method + "/result_" + attack_method
    # path = '../results/ciao_DegreeSAD/' + attack_method + "/result_" + attack_method
    result = pd.read_csv(path, sep='\t', names=names, engine='python')
    p = result.precision.values.tolist()
    r = result.recall.values.tolist()
    n = [attack_name[i]] * len(r)
    P.extend(p)
    R.extend(r)
    N.extend(n)
    # pre_results[attack_name[i]] =p
    # recall_results[attack_name[i]] =r
data_pre = pd.DataFrame({"method": N, "precision": P, "recall": R})
# data_pre = pd.DataFrame(pre_results)
data_pre.boxplot(column='precision', by=['method'])
plt.title("Attack Detection")
plt.ylabel("precision", )
plt.xlabel("Attack Method")
plt.show()
a = 1
#