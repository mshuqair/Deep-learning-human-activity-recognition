from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def plot_class_distribution(y_true):
    values, counts = np.unique(y_true, return_counts=True)
    values = values.astype(int)
    counts = counts.astype(int)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), layout='constrained')
    ax.set_title('Class Distribution')
    ax.pie(x=counts, labels=labels_list, autopct='%.0f%%')
    plt.show()


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    print('Classification metrics:')
    print('Accuracy %.2f, Balanced Accuracy %.2f, F1-score %.2f, Precision %.2f, Recall % .2f'
          % (accuracy, balanced_accuracy, f1, precision, recall))
    print('')


def plot_confusion_matrix(y_true, y_pred):
    display = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred,
                                                      display_labels=labels_list,
                                                      cmap='viridis', colorbar=True,
                                                      xticks_rotation=45)
    display.ax_.set(title='Confusion matrix')
    plt.tight_layout()
    plt.show()


# Main Code
sns.set_theme(style='dark', palette='deep')

labels_list = ['standing', 'sitting', 'lying', 'walking', 'climbing stairs', 'cycling',  'running']
# Read predictions
FileName = 'model_cnn_lstm'
with open('./output/' + FileName + '/model_predictions.pkl', 'rb') as file:
    [y, y_predicted, p_num] = pickle.load(file)

y = y.astype(int)
y_predicted = y_predicted.astype(int)
p_num = p_num.astype(int)

plot_class_distribution(y_true=y)
calculate_metrics(y_true=y, y_pred=y_predicted)
plot_confusion_matrix(y_true=y, y_pred=y_predicted)

