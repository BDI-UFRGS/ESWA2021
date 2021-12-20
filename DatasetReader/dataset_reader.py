import pandas as pd
import numpy as np
np.random.seed(1)
def get_elegible_indexes(df, target_class_index, n_classes):
    value_counts = df[df.columns[target_class_index]].value_counts()
    
    elegible_indexes = value_counts.index[0:n_classes]

    return elegible_indexes


def remove_duplicates(item, subset):
    return item.drop_duplicates(subset=[item.columns[i] for i in subset], keep='first')


def save_dateset_csv(dataset, name):
    dataset_file = open('%s.csv' % name, 'w+')
    dataset_file.write(dataset.to_csv())
    dataset_file.close()


def remove_non_elegible_classes(df, target_class_index, n_classes, subset):  
    elegible_indexes = get_elegible_indexes(df, target_class_index, n_classes)

    dataset = pd.DataFrame()

    print('Initial dataset size: %s' % len(df))

    for index in elegible_indexes:
        item = df[df[df.columns[target_class_index]] == index]

        item = remove_duplicates(item, subset)

        dataset = pd.concat([dataset, item])
        
    dataset = remove_duplicates(dataset, subset)
    
    print('Dataset size after duplicates removal: %s' % len(dataset))

    # save_dateset_csv(dataset, 'dataset')

    return dataset


def read(path, sep, target_class_index, n_classes, subset=[2]):
    df = pd.read_csv(path, sep=sep, engine='python')

    df=df.dropna()
    
    dataset = remove_non_elegible_classes(df, target_class_index, n_classes, subset)

    return dataset


def class_names(dataset, target_class_index):
    y = dataset.iloc[:, target_class_index]
    y = y.astype(str)

    names = np.unique(y)

    return names

def dataset_size(dataset, target_class_index):
    return dataset[dataset.columns[target_class_index]].value_counts()
