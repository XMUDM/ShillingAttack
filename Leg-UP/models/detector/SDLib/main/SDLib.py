import sys

# sys.path.append("../")
from re import split
from models.detector.SDLib.tool.config import Config, LineConfig
from models.detector.SDLib.tool.dataSplit import *
from models.detector.SDLib.tool.file import FileIO


class SDLib(object):
    def __init__(self, config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.relation = []
        self.measure = []
        self.config = config
        self.ratingConfig = LineConfig(config['ratings.setup'])
        self.labels = FileIO.loadLabels(config['label'])

        if self.config.contains('evaluation.setup'):
            self.evaluation = LineConfig(config['evaluation.setup'])

            if self.evaluation.contains('-testSet'):
                # specify testSet
                self.trainingData = FileIO.loadDataSet(config, config['ratings'])
                self.testData = FileIO.loadDataSet(config, self.evaluation['-testSet'], bTest=True)

            elif self.evaluation.contains('-ap'):
                # auto partition
                self.trainingData = FileIO.loadDataSet(config, config['ratings'])
                self.trainingData, self.testData = DataSplit. \
                    dataSplit(self.trainingData, test_ratio=float(self.evaluation['-ap']))

            elif self.evaluation.contains('-cv'):
                # cross validation
                self.trainingData = FileIO.loadDataSet(config, config['ratings'])
                # self.trainingData,self.testData = DataSplit.crossValidation(self.trainingData,int(self.evaluation['-cv']))

        else:
            print('Evaluation is not well configured!')
            exit(-1)

        if config.contains('social'):
            self.socialConfig = LineConfig(self.config['social.setup'])
            self.relation = FileIO.loadRelationship(config, self.config['social'])
        # print('preprocessing...')

    def execute(self):
        # import the algorithm module
        importStr = 'from models.detector.SDLib.method.' + self.config['methodName'] + ' import ' + self.config['methodName']
        exec(importStr)
        if self.config.contains('social'):
            method = self.config[
                         'methodName'] + '(self.config,self.trainingData,self.testData,self.labels,self.relation)'
        else:
            method = self.config['methodName'] + '(self.config,self.trainingData,self.testData,self.labels)'
        ans = eval(method).execute()
        return [float(i) for i in ans]


def run(measure, algor, order):
    measure[order] = algor.execute()
