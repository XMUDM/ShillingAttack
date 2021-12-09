from __future__ import division
from __future__ import print_function
import sys

sys.path.append("../")
import os, argparse, time, math
import numpy as np
import tensorflow as tf
from glob import glob
from utils.attack.data_to_file import *
from test_main.utils_dcgan import *
from numpy import linalg as la
from model.trainer_rec import *
from test_main.main_eval_attack import eval_attack
import utils as ut

flags = tf.app.flags
flags.DEFINE_integer("epoch", 64, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("max_to_keep", 1, "maximum number of checkpoints to keep")
flags.DEFINE_integer("z_dim", 100, "dimensions of z")
#
flags.DEFINE_integer("T", 10, "adv opt epoch")
flags.DEFINE_integer("K", 5, "top k svd")  # 5
flags.DEFINE_float("alpha", 50.0, "opt param")
flags.DEFINE_float("eta", 100.0, "opt param")
flags.DEFINE_integer("attack_num", 50, "attack_num")
flags.DEFINE_integer("filler_num", 90, "filler_num")
FLAGS = flags.FLAGS

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
data_set_name = 'ml100k'
target_ids = [62, 1077, 785, 1419, 1257, 1319, 1612, 1509, 1545, 1373]

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
path_train = '../data/data/' + data_set_name + '_train.dat'
path_test = '../data/data/' + data_set_name + '_test.dat'
attack_info_path = ["../data/data/" + data_set_name + "_selected_items",
                    "../data/data/" + data_set_name + "_target_users"]
# 读取seletced items和target users
attack_info = load_attack_info(*attack_info_path)
dataset_class = ut.load_data.load_data.load_data(path_train=path_train, path_test=path_test,
                                                 header=['user_id', 'item_id', 'rating'],
                                                 sep='\t', print_log=False)


def train_Rec_model(injected_path, injected_profiles, target_id, model_path, train_epoch,
                    model_name='IAutoRec', warm_start=False, restore_path=None):
    tf.reset_default_graph()

    attacked_file_writer(path_train, injected_path, injected_profiles, dataset_class.n_users)

    dataset_class_injected = ut.load_data.load_data.load_data(path_train=injected_path,
                                                              path_test=path_test,
                                                              header=['user_id', 'item_id', 'rating'],
                                                              sep='\t', print_log=False)

    # tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        rec_model = get_model_network(sess, model_name, dataset_class_injected, train_epoch)
        if warm_start:
            # print('warm start')
            rec_model.restore(restore_path)
        rec_model.execute()
        rec_model.save(model_path)
        predictions, hit_ratios = pred_for_target(rec_model, target_id)
    return predictions, hit_ratios


def opt_adv_intent(fake_users, filler_indicators, target_id):
    target_users = attack_info[target_id][1]
    model_path = "./IAutoRec_dcgan_%d.ckpt" % target_id
    injected_path = "./IAutoRec_dcgan_%d.dat" % target_id

    # ----------------------
    for t in range(FLAGS.T):

        injected_profiles = fake_users * filler_indicators
        predictions, _ = train_Rec_model(injected_path, injected_profiles, target_id, model_path, 10)
        f_adv_0 = np.sum(predictions[target_users])
        f_adv_k = f_adv_0
        print("opt_adv_intent\tepoch-%d adv goal\t%f" % (t, f_adv_k))

        delta_f_Adv = []
        B, Sigma, V = la.svd(fake_users)
        for k in range(FLAGS.K):

            Z_k = np.matmul(np.reshape(B[k], [FLAGS.attack_num, 1]), np.reshape(V[k], [1, dataset_class.n_items]))

            fake_users_k = fake_users + FLAGS.alpha * Z_k

            injected_profiles = fake_users_k * filler_indicators
            predictions, _ = train_Rec_model(injected_path, injected_profiles, target_id, model_path,
                                             5, warm_start=True, restore_path=model_path)
            f_adv_k_new = np.sum(predictions[target_users])

            delta_f_Adv.append((f_adv_k_new - f_adv_k) * Z_k)

        delta_f_A = FLAGS.alpha * sum(delta_f_Adv)
        fake_users += FLAGS.eta * delta_f_A
        fake_users[fake_users <= 0] = 0.5
        fake_users[fake_users > 5] = 5
    return fake_users * filler_indicators



tf.reset_default_graph()
with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(sess, dataset_class)
    # print("build_model_ok")
    dcgan.train(FLAGS)
    # save model
    saver = tf.train.Saver()
    saver.save(sess, './dcgan.ckpt')

    fake_users = None
    while True:
        batch_z = gen_random(size=[FLAGS.batch_size, dcgan.z_dim]).astype(np.float32)
        fake_users_ = sess.run(dcgan.G, feed_dict={dcgan.z: batch_z})
        # reshape&[-1,1]->[0,5]
        fake_users_ = fake_users_.reshape([fake_users_.shape[0], -1])
        fake_users_ = (fake_users_ * 2.5) + 2.5
        fake_users = fake_users_ if fake_users is None else np.concatenate([fake_users_, fake_users_], 0)
        if fake_users.shape[0] >= FLAGS.attack_num: break
    # attack_num
    fake_users = fake_users[:FLAGS.attack_num]
    # filler_num
    filler_indicators = []
    for i in range(FLAGS.attack_num):
        fillers_ = np.random.choice(list(range(dataset_class.n_items)), FLAGS.filler_num, replace=False)
        filler_indicator_ = [1 if iid in fillers_ else 0 for iid in range(dataset_class.n_items)]
        filler_indicators.append(filler_indicator_)
    filler_indicators = np.array(filler_indicators)
np.save('./fake_user_dcgan', [fake_users, filler_indicators])
# fake_users, filler_indicators = np.load('./fake_user_dcgan.npy')

results = {}
for target_id in target_ids:

    injected_profiles = opt_adv_intent(fake_users, filler_indicators, target_id)


    model_path = "./IAutoRec_dcgan_%d.ckpt" % target_id
    injected_path = "../data/data/ml100k_%d_dcgan_50_90.dat" % target_id
    target_users = attack_info[target_id][1]
    predictions, hit_ratios = train_Rec_model(injected_path, injected_profiles, target_id, model_path, 500)
    dst_path = "../result/pred_result/" + '_'.join(['IAutoRec', 'ml100k', str(target_id), 'dcgan'])
    target_prediction_writer(predictions, hit_ratios, dst_path)

    result = eval_attack('ml100k', 'IAutoRec', 'dcgan', target_id)
    results[target_id] = result
    print(target_id, result, '\n\n')
    break

for target_id in target_ids:
    print(target_id, results[target_id])
