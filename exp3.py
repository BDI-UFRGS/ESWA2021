from scipy.sparse import data
import DatasetReader as dr
import pandas as pd
from DatasetReader import dataset_reader, dataset_spliter
from DatasetReader import dataset_sampling
from Model import FNN_RNN, FNN, SVM, GaussianNaiveBayes, LogisticRegression, RandomForest, DecisionTree, BernoulliNaiveBayes
from Result import roc_curve, result
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

y_test_p2 = []
fnn_rnn_p2 = []


def run_experiment_p2(path, target_class_index, n_classes, subset, exp, downsample=True):
    print("Experiment: %s" % exp)
    dataset = dataset_reader.read(path=path, sep=';', target_class_index=target_class_index, n_classes=n_classes, subset=subset)
    print(dataset_reader.dataset_size(dataset, target_class_index))

    if downsample:
        dataset, newdataset = dataset_sampling.downsample(dataset, target_class_index)
    print(dataset_reader.dataset_size(dataset, target_class_index))
    class_names = dataset_reader.class_names(dataset, target_class_index)

    f = 1
    for X_train, y_train, X_test, y_test in dataset_spliter.split_train_test(dataset, target_class_index, 10):
        print("Fold: %s" % f)
        f+=1
        X_val, y_val = dataset_spliter.split_x_y(newdataset, target_class_index)
        X_test = pd.concat([X_test, X_val])
        y_test = pd.concat([y_test, y_val])
        print('Train instances: %s' % len(y_train))
        print('Test instances: %s' % len(y_test))
        y_test_p2.append(y_test)

        main_fnn_rnn_predictions, train_time, test_time = FNN_RNN.run(X_train, y_train, X_test, n_classes, epochs=10, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'main-fnn-rnn', path.replace('/', '-')), y_test, main_fnn_rnn_predictions, class_names, train_time, test_time, len(y_train), len(y_test))
        fnn_rnn_p2.append(main_fnn_rnn_predictions)

    result.save_confusion_matrix_all('1-p2', y_test_p2, fnn_rnn_p2, class_names)


def run_experiment(path, target_class_index, n_classes, subset, exp, downsample=True):
    print("Experiment: %s" % exp)
    dataset = dataset_reader.read(path=path, sep=';', target_class_index=target_class_index, n_classes=n_classes, subset=subset)
    print(dataset_reader.dataset_size(dataset, target_class_index))

    if downsample:
        dataset, newdataset = dataset_sampling.downsample(dataset, target_class_index)
    print(dataset_reader.dataset_size(dataset, target_class_index))
    class_names = dataset_reader.class_names(dataset, target_class_index)
    f = 1
    for X_train, y_train, X_test, y_test in dataset_spliter.split_train_test(dataset, target_class_index, 10):
        print("Fold: %s" % f)
        f+=1
        X_val, y_val = dataset_spliter.split_x_y(newdataset, target_class_index)
        X_test = pd.concat([X_test, X_val])
        y_test = pd.concat([y_test, y_val])
        print('Train instances: %s' % len(y_train))
        print('Test instances: %s' % len(y_test))

        y_test_all.append(y_test)

        bnb_predictions, train_time, test_time = BernoulliNaiveBayes.run(X_train, y_train, X_test, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'bnb', path.replace('/', '-')), y_test, bnb_predictions, class_names, train_time, test_time, len(y_train), len(y_test))
        
        gnb_predictions, train_time, test_time = GaussianNaiveBayes.run(X_train, y_train, X_test, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'gnb', path.replace('/', '-')), y_test, gnb_predictions, class_names, train_time, test_time, len(y_train), len(y_test))

        dt_predictions, train_time, test_time = DecisionTree.run(X_train, y_train, X_test, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'dt', path.replace('/', '-')), y_test, dt_predictions, class_names, train_time, test_time, len(y_train), len(y_test))

        rf_predictions, train_time, test_time = RandomForest.run(X_train, y_train, X_test, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'rf', path.replace('/', '-')), y_test, rf_predictions, class_names, train_time, test_time, len(y_train), len(y_test))

        lr_predictions, train_time, test_time = LogisticRegression.run(X_train, y_train, X_test, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'lr', path.replace('/', '-')), y_test, lr_predictions, class_names, train_time, test_time, len(y_train), len(y_test))

        # svm_prediction, train_time, test_time = SVM.run(X_train, y_train, X_test, proba=True)
        # result.save_results('%s-%s-%s' % (exp, 'svm', path.replace('/', '-')), y_test, svm_prediction, class_names, train_time, test_time, len(y_train), len(y_test))

        fnn_predictions, train_time, test_time = FNN.run(X_train, y_train, X_test, n_classes, epochs=10, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'fnn', path.replace('/', '-')), y_test, fnn_predictions, class_names, train_time, test_time, len(y_train), len(y_test))

        fnn_rnn_predictions, train_time, test_time = FNN_RNN.run(X_train, y_train, X_test, n_classes, epochs=10, proba=True)
        result.save_results('%s-%s-%s' % (exp, 'p1', path.replace('/', '-')), y_test, fnn_rnn_predictions, class_names, train_time, test_time, len(y_train), len(y_test))

        bnb.append(bnb_predictions)
        gnb.append(gnb_predictions)
        dt.append(dt_predictions)
        rf.append(rf_predictions)
        lr.append(lr_predictions)
        # svm.append(svm_prediction)
        fnn.append(fnn_predictions)
        fnn_rnn.append(fnn_rnn_predictions)
    
    result.save_confusion_matrix_all('1-bnb', y_test_all, bnb, class_names)
    result.save_confusion_matrix_all('1-gnb', y_test_all, gnb, class_names)
    result.save_confusion_matrix_all('1-dt', y_test_all, dt, class_names)
    result.save_confusion_matrix_all('1-rf', y_test_all, rf, class_names)
    result.save_confusion_matrix_all('1-lr', y_test_all, lr, class_names)
    # result.save_confusion_matrix_all('1-svm', y_test_all, svm, class_names)
    result.save_confusion_matrix_all('1-fnn', y_test_all, fnn, class_names)
    result.save_confusion_matrix_all('1-p1', y_test_all, fnn_rnn, class_names)

    


def run(dataset):
    run_experiment('%sfasttext-crawl-300d-2M.csv' % dataset, 1, 30, [1], '1', downsample=True)


    run_experiment_p2('%sfasttext-crawl-300d-2M.csv' % dataset, 1, 30, [1, 2], '1', downsample=True)


    roc_curve.plot_mean("ROC curve of the Experiment 1", [('Bernoulli Naive Bayes', y_test_all, bnb), 
                                     ('Gaussian Naive Bayes',y_test_all, gnb),
                                     ('Decision Tree', y_test_all, dt),
                                     ('Random Forest', y_test_all, rf),
                                     ('Logistic Regression',y_test_all, lr),
                                    #  ('SVM',y_test_all, svm),
                                     ('FNN',y_test_all, fnn),
                                     ('P1 (Proposed)',y_test_all, fnn_rnn),
                                     ('P2 (Proposed)',y_test_p2, fnn_rnn_p2)])
