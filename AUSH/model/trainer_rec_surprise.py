# -*- coding: utf-8 -*-
# @Time       : 2019/8/23 15:24
# @Author     : chensi
# @File       : cf.py
# @Software   : PyCharm
# @Desciption : None

import os
from surprise import Dataset, Reader, accuracy
from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise.model_selection import PredefinedKFold
from collections import defaultdict


def get_top_n(predictions, n=50):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


def get_model(model_name):
    algo = None
    if 'KNN' in model_name:
        model_name = model_name.split('_')
        knn_model_name = model_name[0]
        user_based = False if len(model_name) > 1 and model_name[1] == 'I' else True
        dis_method = 'msd' if len(model_name) < 3 else model_name[2]
        k = 20 if len(model_name) < 4 else int(model_name[3])
        sim_options = {'user_based': user_based, 'name': dis_method}
        if knn_model_name == 'KNNBasic':
            algo = KNNBasic(sim_options=sim_options, k=k)
        elif knn_model_name == 'KNNWithMeans':
            algo = KNNWithMeans(sim_options=sim_options, k=k)
        elif knn_model_name == 'KNNWithZScore':
            algo = KNNWithZScore(sim_options=sim_options, k=k)
    elif 'SVDpp' in model_name or 'SVD' in model_name or 'NMF' in model_name:
        model_name = model_name.split('_')
        n_factors = 25 if len(model_name) == 1 else int(model_name[1])
        if model_name[0] == 'SVDpp':
            algo = SVDpp(n_factors=n_factors)
        elif model_name[0] == 'SVD':
            algo = SVD(n_factors=n_factors)
        elif model_name[0] == 'NMF':
            algo = NMF(n_factors=n_factors)
    return algo


def get_model_old(model_name):
    algo = None
    if model_name == 'KNNBasic_U':
        sim_options = {'user_based': True}
        algo = KNNBasic(sim_options=sim_options, k=20)
    elif model_name == 'KNNBasic_I':
        sim_options = {'user_based': False}
        algo = KNNBasic(sim_options=sim_options, k=20)
        # algo = KNNBasic()
    elif model_name == 'KNNWithMeans_I':
        algo = KNNWithMeans(sim_options={'user_based': False}, k=20)
    elif model_name == 'KNNWithMeans_U':
        algo = KNNWithMeans(sim_options={'user_based': True}, k=20)
    elif model_name == 'KNNWithZScore_I':
        algo = KNNWithZScore(sim_options={'user_based': False}, k=20)
    elif model_name == 'KNNWithZScore_U':
        algo = KNNWithZScore(sim_options={'user_based': True}, k=20)
    elif model_name == 'SVDpp':
        algo = SVDpp()
    elif model_name == 'SVD':
        algo = SVD()
    elif model_name == 'NMF':
        algo = NMF()
    elif 'NMF_' in model_name:
        n_factors = int(model_name.split("_")[1])
        algo = NMF(n_factors=n_factors)
    elif 'SVDpp_' in model_name:
        n_factors = int(model_name.split("_")[1])
        algo = SVDpp(n_factors=n_factors)
    elif 'SVD_' in model_name:
        n_factors = int(model_name.split("_")[1])
        algo = SVD(n_factors=n_factors)
    elif 'KNNBasic_U_' in model_name:
        k = int(model_name.split("_")[-1])
        sim_options = {'user_based': True}
        algo = KNNBasic(sim_options=sim_options, k=k)
    elif 'KNNBasic_I_' in model_name:
        k = int(model_name.split("_")[-1])
        sim_options = {'user_based': False}
        algo = KNNBasic(sim_options=sim_options, k=k)
    return algo


def basic_rec(model_name, train_path, test_path, target_id):
    # build data
    # TODO check float and min_r
    reader = Reader(line_format='user item rating', sep='\t', rating_scale=(1, 5))
    data = Dataset.load_from_folds([(train_path, test_path)], reader=reader)
    trainset, testset = None, None
    pkf = PredefinedKFold()
    for trainset_, testset_ in pkf.split(data):
        trainset, testset = trainset_, testset_

    # train model
    rec_algo = get_model(model_name)
    rec_algo.fit(trainset)
    # eval
    preds = rec_algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=True)

    # predor target
    fn_pred = lambda uid: rec_algo.predict(str(uid), str(target_id), r_ui=0).est
    target_predictions = list(map(fn_pred, range(trainset.n_users)))

    # topn
    testset = trainset.build_anti_testset()
    predictions = rec_algo.test(testset)
    top_n = get_top_n(predictions, n=50)

    hit_ratios = {}
    for uid, user_ratings in top_n.items():
        topN = [int(iid) for (iid, _) in user_ratings]
        hits = [1 if target_id in topN[:i] else 0 for i in [1, 3, 5, 10, 20, 50]]
        hit_ratios[int(uid)] = hits
    return target_predictions, hit_ratios
