#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ariaschen
# datetime:2020/1/14 09:11
# software: PyCharm

# import itertools, gzip
import pandas as pd


columns = ['Rec_model', 'attack_method', 'target_id']

hr = ['HR_1', 'HR_3', 'HR_5', 'HR_10', 'HR_20', 'HR_50']
hr_ori = [i + '_ori' for i in hr]

columns += [i + '_inseg' for i in ['shift'] + hr_ori + hr]

columns += [i + '_all' for i in ['shift'] + hr_ori + hr]

columns_r = [i + '_inseg' for i in ['shift'] + hr] + [i + '_all' for i in ['shift'] + hr]
""""""
# data = pd.read_excel('filmTrust_distance.xls')
# data.groupby('attack_method').mean()[['dis_TVD','dis_JS']].to_excel('filmTrust_distance_avg.xls')

# data = pd.read_excel('ml100k_performance_all.xls')
# data = pd.read_excel('../result_ijcai/filmTrust_performance_all.xls')
# data = pd.read_excel('../result_ijcai/ml100k_performance_all.xls')
# data = pd.read_excel('office_performance_all.xls')
data = pd.read_excel('automotive_performance_all.xls')
data.columns = columns
data = data[['Rec_model', 'attack_method', 'target_id', 'shift_inseg', 'HR_10_inseg', 'shift_all', 'HR_10_all']]
# target_type_dict = dict(
#     zip([62, 1077, 785, 1419, 1257] + [1319, 1612, 1509, 1545, 1373], ['random'] * 5 + ['tail'] * 5))
# target_type_dict = dict(zip([5, 395, 181, 565, 254] + [601, 623, 619, 64, 558], ['random'] * 5 + ['tail'] * 5))
target_type_dict = dict(zip([1141, 1656, 477, 1089, 866] + [88, 22, 122, 339, 1431], ['random'] * 5 + ['tail'] * 5))
data['target_type'] = data.target_id.apply(lambda x: target_type_dict[x])
data['attack_method'] = data.attack_method.apply(lambda x: x.split('_')[0])
result = data.groupby(['Rec_model','attack_method', 'target_type']).mean()[['shift_all', 'HR_10_all']]
result.to_excel('ml100k_performance_0119_sample_strategy.xlsx')
exit()
