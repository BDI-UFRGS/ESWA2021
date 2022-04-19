import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def get_difference(y_test, prediction):
    p = []
    for i in range(len(prediction)):
        if y_test[i] == 0:
            p.append(1 - prediction[i][0])
        else:
            p.append(prediction[i][1])
    return p


def plot_mean(exp, models):
    plt.figure(figsize=(10,6))

    for name, y, p in models:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i in range(len(y)):
            y_test = y[i]
            y_test = [np.argmax(t, axis=0) for t in np.asarray(y_test)]

            prediction = p[i]
            prediction = get_difference(y_test, prediction)
            fpr, tpr, _ = roc_curve(y_test, prediction, pos_label=1)
            roc_auc = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, lw=1, label='Mean ROC of %s model (AUC = %0.2f $\pm$ %0.2f)' % (name, mean_auc, std_auc))
  
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s' % exp.capitalize())
    plt.legend(loc="lower right")
    plt.savefig('../results/%s.png' % exp, bbox_inches='tight', dpi=1000)

def plot(y_test, predictions):
    y_test = [np.argmax(t, axis=0) for t in np.asarray(y_test)]
    plt.figure()
    for name, prediction in predictions:
        prediction = get_difference(y_test, prediction)
        print(len(y_test))
        print(len(prediction))
        fpr, tpr, _ = roc_curve(y_test, prediction, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='%s (area = %0.2f)' % (name, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


