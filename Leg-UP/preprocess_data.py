import os
import numpy as np
import random
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

data_file = Path('data')
data_set_name = 'GroceryFood'
data_file = data_file / data_set_name / (data_set_name + 'Raw.json')

df_gro = pd.DataFrame(columns=['user', 'item', 'score'])
data = []
with open(data_file, encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
print(f'data_set_len:{len(data)}')
print(f'data head:\n{data[:5]}')

user_set = set()
item_set = set()
data_list = []
for idx, d in enumerate(data):
    item_set.add(d['asin'])
    user_set.add(d["reviewerID"])
    data_list.append([d["reviewerID"], d['asin'], d['overall']])

raw_df = pd.DataFrame(data_list, columns=['user', 'item', 'score'])

user2idx = {x: idx for idx, x in enumerate(user_set)}
item2idx = {x: idx for idx, x in enumerate(item_set)}


def fun(item):
    return user2idx[item]

def fun2(item):
    return item2idx[item]

raw_df['user'] = raw_df['user'].apply(fun)
raw_df['item'] = raw_df['item'].apply(fun2)

print(f'raw data frame:')
print(raw_df)

user_cont = raw_df.groupby('user').count()
filter_ratings = {i for i in list(user_cont[user_cont['item'] >= 17].index)}

after_filter_df = pd.DataFrame(columns=['user', 'item', 'score'])


all_data = []
for i in filter_ratings:
    each_i = raw_df[raw_df['user'] == i]
    all_data.append(each_i.values)
    after_filter_df = after_filter_df.append(each_i)

train_list = []
test_list = []
train_df = pd.DataFrame(columns=['user', 'item', 'score'])
test_df = pd.DataFrame(columns=['user', 'item', 'score'])
for d in all_data:
    train, test = train_test_split(d, test_size=0.1, random_state=42)
    df = pd.DataFrame(train, columns=['user', 'item', 'score'])
    df2 = pd.DataFrame(test, columns=['user', 'item', 'score'])
    train_df = train_df.append(df)
    test_df = test_df.append(df2)
print(f'train_df:{train_df}')
print(f'test_df:{test_df}')

item_count = raw_df.groupby('item').count().sort_values(by='user', ascending=False)
print(item_count)
target_item_first = [i for i in item_count[:int(0.1 * len(item_count))].index.values]
target_item_last = [i for i in item_count[int(0.9 * len(item_count)):].index.values]
target_item = target_item_first + target_item_last
with open(f'data/{data_set_name}_target_item', 'w') as f:
    for i in target_item:
        f.write(str(int(i)))
        f.write('\n')

with open(f'data/{data_set_name}_selected_items', 'a+') as f:
    for i in target_item:
        select_item = [i]
        while True:
            a = random.choice(target_item_first)
            if a not in select_item:
                select_item.append(a)
            if len(select_item) == 4:
                break
        f.write(str(select_item[0]) + '\t')
        f.write(str(select_item[1]) + ',' + str(select_item[2]) + ',' + str(select_item[3]))
        f.write('\n')



user_cont = raw_df.groupby('user').count()
filter_ratings = {i for i in list(user_cont[user_cont['item'] >= 17].index)}

after_filter_df = pd.DataFrame(columns=['user', 'item', 'score'])

all_data = []
for i in filter_ratings:
    each_i = raw_df[raw_df['user'] == i]
    all_data.append(each_i.values)
    after_filter_df = after_filter_df.append(each_i)

# all_data = []
# for i in filter_ratings:
#     each_i = raw_df[raw_df['user'] == i]
#     all_data.append(each_i.values)
#     after_filter_df = after_filter_df.append(each_i)



# dfv = train_df.values
# print(dfv)
# with open(f'data/{data_set_name}_train.dat', 'a', encoding='utf-8') as f:
#     for d in dfv:
#         for idx, i in enumerate(d):
#             if idx != 2:f.write(str(int(i)))
#             else : f.write(str(i))
#             if idx != 2: f.write('\t')
#         f.write('\n')
#         dfv = train_df.values
#
# dfv = test_df.values
# with open(f'data/{data_set_name}_test.dat', 'a', encoding='utf-8') as f:
#     for d in dfv:
#         for idx, i in enumerate(d):
#             if idx != 2: f.write(str(int(i)))
#             else: f.write(str(i))
#             if idx != 2: f.write('\t')
#         f.write('\n')
