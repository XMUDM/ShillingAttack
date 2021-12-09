# -*- coding: utf-8 -*-
# @Time       : 2019/8/23 11:53
# @Author     : chensi
# @File       : load_attack_info.py
# @Software   : PyCharm
# @Desciption : None

def load_attack_info(seletced_item_path, target_user_path):
    attack_info = {}
    with open(seletced_item_path, "r") as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            target_item, selected_items = int(line[0]), list(map(int, line[1].split(",")))
            attack_info[target_item] = [selected_items]
    with open(target_user_path, "r") as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            target_item, target_users = int(line[0]), list(map(int, line[1].split(",")))
            attack_info[target_item].append(target_users)
    return attack_info

