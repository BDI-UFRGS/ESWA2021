import pandas as pd
import numpy as np
np.random.seed(1)
def get_elegible_indexes(df, n_classes):
    value_counts = df[df.columns[0]].value_counts()
    
    elegible_indexes = value_counts.index[0:n_classes]

    return elegible_indexes


def remove_duplicates(item, subset):
    return item.drop_duplicates(subset=[item.columns[i] for i in subset], keep='last')


def save_dateset_csv(dataset, name):
    dataset_file = open('%s.csv' % name, 'w+')
    dataset_file.write(dataset.to_csv())
    dataset_file.close()


def remove_non_elegible_classes(df, n_classes, subset):  
    elegible_indexes = get_elegible_indexes(df, n_classes)

    dataset = pd.DataFrame()

    print('Initial dataset size: %s' % len(df))

    for index in elegible_indexes:
        item = df[df[df.columns[0]] == index]

        item = remove_duplicates(item, subset)

        dataset = pd.concat([dataset, item])
        
    dataset = remove_duplicates(dataset, subset)
    
    print('Dataset size after duplicates removal: %s' % len(dataset))

    # save_dateset_csv(dataset, 'dataset')

    return dataset


def read(path, sep, n_classes, subset):
    df = pd.read_csv(path, sep=sep, engine='python')

    df=df.dropna()
    
    dataset = remove_non_elegible_classes(df, n_classes, subset)

    return dataset


def class_names(dataset):
    y = dataset.iloc[:, 0]
    y = y.astype(str)

    names = np.unique(y)

    return names

def dataset_size(dataset):
    return dataset[dataset.columns[0]].value_counts()
