import pandas as pd
import numpy as np

np.random.seed(1)


def get_minority_class_size(dataset, target_class_index):
    value_counts = dataset[dataset.columns[target_class_index]].value_counts()
    return min([dataset[dataset[dataset.columns[target_class_index]] == index].shape[0] for index in value_counts.index])


def downsample(dataset, target_class_index):
    value_counts = dataset[dataset.columns[target_class_index]].value_counts()

    min_class_size = get_minority_class_size(dataset, target_class_index) - 1

    new_dataset = pd.DataFrame()
    new_dataset_outsamples = pd.DataFrame()
    print('aqui')
    print(value_counts)
    print(min_class_size)
    for index in value_counts.index:
        item = dataset[dataset[dataset.columns[target_class_index]] == index]
        item_shape = item.shape[0]
        all_indexes = pd.Index(np.arange(item_shape))
        chosen_idx = np.random.choice(item_shape, replace=False, size=min_class_size) 
        ignored_indexes = all_indexes.difference(chosen_idx)

        new_dataset = pd.concat([new_dataset, item.iloc[chosen_idx]], axis=0)
        new_dataset_outsamples = pd.concat([new_dataset_outsamples, item.iloc[ignored_indexes]], axis=0)        
        
    return new_dataset, new_dataset_outsamples