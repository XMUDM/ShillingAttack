#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:ariaschen
# datetime:2020/1/12 16:11
# software: PyCharm

import itertools, gzip
import pandas as pd
from utils.load_data.load_data import *
from sklearn.model_selection import train_test_split


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def data_preprocess(data_set, gz_path):
    data = getDF(gz_path)[['reviewerID', 'asin', 'overall']]
    data.columns = ['uid', 'iid', 'rating']

    uids, iids = data.uid.unique(), data.iid.unique()
    n_uids, n_iids, n_ratings = len(uids), len(iids), data.shape[0]
    print('User num:', n_uids, '\tItem num:', n_iids, '\tRating num:', n_ratings, '\t Sparsity :', n_ratings / (n_iids * n_uids))
    print('Number of ratings per user:', n_ratings / n_uids)

    uid_update = dict(zip(uids, range(n_uids)))
    iid_update = dict(zip(iids, range(n_iids)))

    data.uid = data.uid.apply(lambda x: uid_update[x])
    data.iid = data.iid.apply(lambda x: iid_update[x])

    train_idxs, test_idxs = train_test_split(list(range(n_ratings)), test_size=0.1)

    train_data = data.iloc[train_idxs]
    test_data = data.iloc[test_idxs]
    path_train = "../data/data/" + data_set + "_train.dat"
    path_test = "../data/data/" + data_set + "_test.dat"
    train_data.to_csv(path_train, index=False, header=None, sep='\t')
    test_data.to_csv(path_test, index=False, header=None, sep='\t')
    np.save("../data/data/" + data_set + "_id_update", [uid_update, iid_update])


def exp_select(data_set, target_items, selected_num, target_user_num):
    path_test = "../data/data/" + data_set + "_test.dat"
    path_train = "../data/data/" + data_set + "_train.dat"
    dataset_class = load_data(path_train=path_train, path_test=path_test,
                              header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)

    item_pops = dataset_class.get_item_pop()

    items_sorted = np.array(item_pops).argsort()[::-1]

    bandwagon_selected = items_sorted[:selected_num]
    print('bandwagon_selected:', bandwagon_selected)


    threshold = dataset_class.test_data.rating.mean()
    threshold = threshold if threshold < 3 else 3.0
    print('threshold:', threshold)
    selected_candidates = items_sorted[:20]

    selected_candidates = list(itertools.combinations(selected_candidates, selected_num))

    result = {}
    target_items = [j for i in range(2, 10) for j in
                    items_sorted[i * len(items_sorted) // 10:(i * len(items_sorted) // 10) + 2]][::-1]
    target_items = list(
        np.random.choice([i for i in range(len(item_pops)) if item_pops[i] == 3], 4, replace=False)) + target_items
    print('target_items:', target_items)
    print('number of ratings:', [item_pops[i] for i in target_items])
    for target in target_items:
        target_rated = set(dataset_class.train_data[dataset_class.train_data.item_id == target].user_id.values)
        data_tmp = dataset_class.train_data[~dataset_class.train_data.user_id.isin(target_rated)].copy()
        data_tmp = data_tmp[data_tmp.rating >= threshold]
        np.random.shuffle(selected_candidates)

        for selected_items in selected_candidates:
            target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby(
                'user_id').size()

            if target_users[(target_users == selected_num)].shape[0] >= target_user_num:
                target_users = sorted(target_users[(target_users == selected_num)].index)
                result[target] = [sorted(selected_items), target_users]
                print('target:', target)
                break

        if target not in result:
            for selected_items in selected_candidates:

                target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby(
                    'user_id').size()
                target_users = sorted(dict(target_users).items(), key=lambda x: x[1], reverse=True)
                min = target_users[target_user_num][1]
                target_users = [i[0] for i in target_users[:target_user_num] if i[1] > selected_num // 2]
                if len(target_users) >= target_user_num:
                    result[target] = [sorted(selected_items), sorted(target_users)]
                    print('target:', target, 'min rated selected item num：', min)
                    break

        if target not in result:
            print('target:', target, 'non-targeted user')
            a = 1

    key = list(result.keys())
    selected_items = [','.join(map(str, result[k][0])) for k in key]
    target_users = [','.join(map(str, result[k][1])) for k in key]
    selected_items = pd.DataFrame(dict(zip(['id', 'selected_items'], [key, selected_items])))
    target_users = pd.DataFrame(dict(zip(['id', 'target_users'], [key, target_users])))
    selected_items.to_csv("../data/data/" + data_set + '_selected_items', index=False, header=None, sep='\t')
    target_users.to_csv("../data/data/" + data_set + '_target_users', index=False, header=None, sep='\t')


if __name__ == '__main__':
    data_set = 'office'
    gz_path = 'C:\\Users\\ariaschen\\Downloads\\reviews_Office_Products_5.json.gz'
    # data_set = 'automotive'
    # gz_path = 'C:\\Users\\ariaschen\\Downloads\\reviews_Automotive_5.json.gz'
    # data_set = 'grocery'
    # gz_path = "../data/new_raw_data/reviews_Grocery_and_Gourmet_Food_5.json.gz"


    data_preprocess(data_set, gz_path)

    target_items = None

    exp_select(data_set, target_items, selected_num=2, target_user_num=30)
