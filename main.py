from scipy.sparse import data
import DatasetReader as dr
import pandas as pd
from DatasetReader import dataset_reader, dataset_spliter
from DatasetReader import dataset_sampling
from Model import FNN_RNN, FNN, RNN, SVM, GaussianNaiveBayes, LogisticRegression, RandomForest, DecisionTree, BernoulliNaiveBayes
from Plot import roc_curve
import numpy as np





y_test_all = []
bnb = []
gnb = []
dt = []
rf = []
lr = []
svm = []
fnn = []
fnn_rnn = []

def run_experiment(path, target_class_index, n_classes, subset):

    dataset = dataset_reader.read(path=path, sep=';', target_class_index=target_class_index, n_classes=n_classes, subset=subset)
    print(dataset_reader.dataset_size(dataset, target_class_index))

    dataset, newdataset = dataset_sampling.downsample(dataset, target_class_index)
    print(dataset_reader.dataset_size(dataset, target_class_index))
    # print(dataset_reader.dataset_size(newdataset, target_class_index))
    class_names = dataset_reader.class_names(dataset, target_class_index)

    for X_train, y_train, X_test, y_test in dataset_spliter.split_train_test(dataset, target_class_index, 10):
        X_val, y_val = dataset_spliter.split_x_y(newdataset, target_class_index)

        print('Train instances: %s' % len(y_train))
        print('Test instances: %s' % len(y_test))
        print('Val instances: %s' % len(y_val))

        y_test_all.append(y_test)

        bnb_predictions = BernoulliNaiveBayes.run(X_train, y_train, X_test)
        
        gnb_predictions = GaussianNaiveBayes.run(X_train, y_train, X_test)

        dt_predictions = DecisionTree.run(X_train, y_train, X_test)

        rf_predictions = RandomForest.run(X_train, y_train, X_test)

        lr_predictions = LogisticRegression.run(X_train, y_train, X_test)

        svm_prediction = SVM.run(X_train, y_train, X_test)

        fnn_predictions = FNN.run(X_train, y_train, X_test, n_classes, epochs=10)

        # rnn_predictions = RNN.run(X_train, y_train, X_test, n_classes, epochs=3)

        fnn_rnn_predictions = FNN_RNN.run(X_train, y_train, X_test, n_classes, epochs=10)

        bnb.append(bnb_predictions)
        gnb.append(gnb_predictions)
        dt.append(dt_predictions)
        rf.append(rf_predictions)
        lr.append(lr_predictions)
        svm.append(svm_prediction)
        fnn.append(fnn_predictions)
        fnn_rnn.append(fnn_rnn_predictions)

        # roc_curve.plot_roc(y_test, [('Bernoulli Naive Bayes', bnb_predictions), 
        #                             ('Gaussian Naive Bayes', gnb_predictions),
        #                             ('Decision Tree', dt_predictions),
        #                             ('Random Forest', rf_predictions),
        #                             ('Logistic Regression', lr_predictions),
        #                             ('SVM', svm_prediction),
        #                             ('FNN', fnn_predictions),
        #                             ('FNN+RNN (Proposed)', fnn_rnn_predictions)])
        



# path = 'dataset/fasttext/crawl-300d-2M/particular.csv'
# target_class_index = 0
# n_classes = 2
# subset = [2]

run_experiment('dataset/fasttext/crawl-300d-2M/particular.csv', 0, 2, [2])
run_experiment('dataset/fasttext/crawl-300d-2M-subword/particular.csv', 0, 2, [2])
run_experiment('dataset/fasttext/wiki-news-300d-1M/particular.csv', 0, 2, [2])
run_experiment('dataset/fasttext/wiki-news-300d-1M-subword/particular.csv', 0, 2, [2])

run_experiment('dataset/glove/glove6B300d/particular.csv', 0, 2, [2])
run_experiment('dataset/glove/glove42B300d/particular.csv', 0, 2, [2])
run_experiment('dataset/glove/glove840B300d/particular.csv', 0, 2, [2])

run_experiment('dataset/word2vec/particular.csv', 0, 2, [2])

roc_curve.plot_mean(y_test_all, [('Bernoulli Naive Bayes', bnb), 
                                ('Gaussian Naive Bayes', gnb),
                                ('Decision Tree', dt),
                                ('Random Forest', rf),
                                ('Logistic Regression', lr),
                                ('SVM', svm),
                                ('FNN', fnn),
                                ('FNN+RNN (Proposed)', fnn_rnn)])
