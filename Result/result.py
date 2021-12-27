import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def save_confusion_matrix_all(name, y_test, predictions, class_names):

    y = []
    p = []

    for y_t in y_test:
        y_t = [np.argmax(t, axis=0) for t in np.asarray(y_t)]
        y = np.concatenate((y, y_t), axis=None)

    for pred in predictions:
        pred = [np.argmax(t, axis=0) for t in np.asarray(pred)]
        p = np.concatenate((p, pred), axis=None)

    print(y)
    print(p)

    save_confusion_matrix(y, p, name, class_names)

def save_results(name, y_test, predictions, class_names, train_time, test_time, train_size, test_size):
    y_test = [np.argmax(t, axis=0) for t in np.asarray(y_test)]
    predictions = [np.argmax(t, axis=0) for t in np.asarray(predictions)]


    f1_micro = f1_score(y_test, predictions, average='micro', labels=np.unique(predictions))
    f1_macro = f1_score(y_test, predictions, average='macro',labels=np.unique(predictions))
    f1_weighted = f1_score(y_test, predictions, average='weighted',labels=np.unique(predictions))
    precision_micro = precision_score(y_test, predictions, average='micro',labels=np.unique(predictions))
    precision_macro = precision_score(y_test, predictions, average='macro',labels=np.unique(predictions))
    precision_weighted = precision_score(y_test, predictions, average='weighted',labels=np.unique(predictions))
    recall_micro = recall_score(y_test, predictions, average='micro',labels=np.unique(predictions))
    recall_macro = recall_score(y_test, predictions, average='macro',labels=np.unique(predictions))
    recall_weighted = recall_score(y_test, predictions, average='weighted',labels=np.unique(predictions))
    results = [f1_micro, f1_macro, f1_weighted, precision_micro,precision_macro,precision_weighted,recall_micro,recall_macro,recall_weighted, train_time, test_time, train_size, test_size]
  
    file_output = open(name + '.txt', 'a')
    file_output.write(name + ';' + ';'.join([str(x) for x in results]) + '\n')
    file_output.close()


def save_confusion_matrix(y_test, predictions, name, class_names):
    conf_mat = confusion_matrix(y_test, predictions)
    group_counts = ['{0:0.0f}'.format(value) for value in
                conf_mat.flatten()]

    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    
    group_percentages = ['{0:.2%}'.format(value) for value in
                     conf_mat.flatten()]


    labels = [f'{v1}\n{v2}' for v1, v2 in
          zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(len(class_names), len(class_names))

    df_cm = pd.DataFrame(conf_mat, index = [i for i in class_names], columns = [i for i in class_names])
    plt.figure(figsize=(30, 20))
    # plt.figure()
    plt.title('Confusion Matrix of the Experiment %s using the %s model' % (name.split('-')[0].capitalize(), name.split('-')[1].upper()))

    sn.heatmap(df_cm, annot=labels, cmap='Blues', fmt='', vmin=0, vmax=1)
    plt.savefig('%s.png' % name)