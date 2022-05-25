from sklearn.model_selection import StratifiedKFold
import pandas as pd


def split_train_test(dataset, n_folds):
    skf = StratifiedKFold(n_splits=n_folds)
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]
    y = y.astype(str)
    
    for train_index, test_index in skf.split(X, y):
         yield X.iloc[train_index], pd.get_dummies(y.iloc[train_index]), X.iloc[test_index], pd.get_dummies(y.iloc[test_index])

def split_x_y(dataset):
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]
    y = y.astype(str)

    return X, pd.get_dummies(y)