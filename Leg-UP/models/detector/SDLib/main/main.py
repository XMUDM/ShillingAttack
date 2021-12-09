import sys

sys.path.append("../")
from SDLib import SDLib
from tool.config import Config

if __name__ == '__main__':

    print('=' * 80)
    print('   SDLib: A Python library used to collect shilling detection methods.')
    print('=' * 80)
    print('Supervised Methods:')
    print('1. DegreeSAD   2.CoDetector   3.BayesDetector\n')
    print('Semi-Supervised Methods:')
    print('4. SemiSAD\n')
    print('Unsupervised Methods:')
    print('5. PCASelectUsers    6. FAP   7.timeIndex\n')
    print('-' * 80)
    algor = -1
    conf = -1
    order = 6  # input('please enter the num of the method to run it:')
    import time

    s = time.clock()
    # if order == 0:
    #     try:
    #         import seaborn as sns
    #     except ImportError:
    #         print '!!!To obtain nice data charts, ' \
    #               'we strongly recommend you to install the third-party package <seaborn>!!!'
    #     conf = Config('../config/visual/visual.conf')
    #     Display(conf).render()
    #     exit(0)

    if order == 1:
        conf = Config('../config/DegreeSAD_tmp.conf')

    elif order == 2:
        conf = Config('../config/CoDetector.conf')

    elif order == 3:
        conf = Config('../config/BayesDetector.conf')

    elif order == 4:
        conf = Config('../config/SemiSAD.conf')

    elif order == 5:
        conf = Config('../config/PCASelectUsers.conf')

    elif order == 6:
        conf = Config('../config/FAP.conf')
    elif order == 7:
        conf = Config('../config/timeIndex.conf')

    else:
        print('Error num!')
        exit(-1)

    # ori conf info
    lines = []
    with open('../config/FAP.conf', 'r') as fin:
        for line in fin:
            lines.append(line)
    random = [5, 395, 181, 565, 254]
    tail = [601, 623, 619, 64, 558]
    targets = random + tail
    # targets = [62, 1077, 785, 1419, 1257] + [1319, 1612, 1509, 1545, 1373]
    attack_methods = ["segment", "average", "random", "bandwagon", "gan"]
    for attack_method in attack_methods[0:]:
        for iid in targets:
            path = "../dataset/GAN/filmTrust/filmTrust_" + str(iid) + "_" + attack_method + "_50_36.dat"
            # path = "../dataset/GAN/ciao_1/ciao_" + str(iid) + "_" + attack_method + "_50_15.dat"
            lines[0] = 'ratings=' + path + '\n'
            # lines[-1] = "output.setup=on -dir ../results/ciao_DegreeSAD/" + attack_method + '/'
            lines[-1] = "output.setup=on -dir ../results/filmTrust_0903_FAP/" + attack_method + '/'
            with open('../config/FAP_t.conf', 'w') as fout:
                fout.write(''.join(lines))
            sd = SDLib(Config('../config/FAP_t.conf'))
            result = sd.execute()
    # conf = Config('../config/DegreeSAD_t.conf')
    # conf = Config('../config/FAP_t.conf')
    # sd = SDLib(conf)
    # sd.execute()
    e = time.clock()
    print("Run time: %f s" % (e - s))
