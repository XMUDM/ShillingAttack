# -*- coding: utf-8 -*-
# @Time       : 2020/12/27 19:57
# @Author     : chensi
# @File       : run.py
# @Software   : PyCharm
# @Desciption : None


import argparse, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

PythonCommand = 'python'  # 'D:\Anaconda3\envs\py38_tf2\python' if os.path.exists('D:\Anaconda3') else 'python'


class Run:
    def __init__(self):
        self.args = self.parse_args()
        self.args.attacker_list = self.args.attacker_list.split(',')
        self.args.recommender_list = self.args.recommender_list.split(',')

    def execute(self):

        self.step_1_Rec()

        self.step_2_Attack()

        return

    def parse_args(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_set', type=str, default='ml100k')  # ml100k,filmTrust,automotive
        parser.add_argument('--attack_num', type=int, default=50)
        parser.add_argument('--filler_num', type=int, default=36)
        parser.add_argument('--cuda_id', type=int, default=3)
        parser.add_argument('--use_cuda', type=int, default=0)
        parser.add_argument('--batch_size_S', type=int, default=64)
        parser.add_argument('--batch_size_D', type=int, default=64)
        parser.add_argument("--surrogate", type=str, default="WMF")
        

        # ml100k:62,1077,785,1419,1257
        # filmTrust:5,395,181,565,254
        # automotive:119,422,594,884,1593
        parser.add_argument('--target_ids', type=str, default='62')
        # AUSH,AUSHplus,RecsysAttacker,DCGAN,WGAN,SegmentAttacker,BandwagonAttacker,AverageAttacker,RandomAttacker
        parser.add_argument('--attacker_list', type=str, default='AUSHplus')
        # SVD,NMF,SlopeOne,IAutoRec,UAutoRec,NeuMF
        parser.add_argument('--recommender_list', type=str, default='SVD,NMF,SlopeOne,IAutoRec,UAutoRec,NeuMF')
        return parser.parse_args()

    def step_1_Rec(self):
        print('step_1')
        args = self.args
        """

        data_set/target_ids/train_path/test_path/model_path/target_prediction_path_prefix
    
        """
        args_dict = {
            'exe_model_lib': 'recommender',
            'train_path': './data/%s/%s_train.dat' % (args.data_set, args.data_set),
            'test_path': './data/%s/%s_test.dat' % (args.data_set, args.data_set),
        }
        args_dict.update(vars(args))

        #
        for recommender in args.recommender_list:
            #
            cur_args_dict = {
                'exe_model_class': recommender,
                'model_path': './results/model_saved/%s/%s_%s' % (args.data_set, args.data_set, recommender),
                'target_prediction_path_prefix': './results/performance/mid_results/%s/%s_%s' % (
                    args.data_set, args.data_set, recommender),
            }
            cur_args_dict.update(args_dict)

            args_str = ' '.join(
                ["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
            #
            print('%s ./execute_model.py %s' % (PythonCommand, args_str))
            print(os.system('%s ./execute_model.py %s' % (PythonCommand, args_str)))

    def step_2_Attack(self):
        print('step_2')
        args = self.args

        args_dict = {
            'exe_model_lib': 'attacker',
            # 'filler_num': 4,
            # 'epoch': 50
        }
        args_dict.update(vars(args))

        for target_id in map(int, args.target_ids.split(',')):
            for attacker in args.attacker_list:
                cur_args_dict = {
                    'exe_model_class': attacker,
                    'target_id': target_id,
                    'injected_path': './results/data_attacked/%s/%s_%s_%d.data' % (
                        args.data_set, args.data_set, attacker, target_id)

                }
                cur_args_dict.update(args_dict)

                args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in cur_args_dict.items()])
                print(os.system('%s ./execute_model.py %s' % (PythonCommand, args_str)))
            # break

model = Run()
model.execute()


