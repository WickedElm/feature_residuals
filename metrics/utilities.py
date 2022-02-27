#!/usr/bin/env python

import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import ipdb
import sklearn.metrics
import matplotlib.ticker
import sklearn.decomposition
import sklearn.preprocessing

def add_metric_to_db(db_path, experiment_name, dataset, metric_name, metric):
    if not os.path.exists(db_path):
        print(f'WARNING:  Could not find path {db_path}')

    db_filename = f'{db_path}/{metric_name}.csv'
    
    if os.path.exists(db_filename):
        metrics_df = pd.read_csv(db_filename, index_col=0)
        metrics_df.loc[experiment_name, dataset] = metric
    else:
        metrics_df = pd.DataFrame({dataset:metric}, index=[experiment_name])

    metrics_df.to_csv(db_filename, index=True)

def f1_score(predicted_labels=None, true_labels=None, metrics_dir='.', is_training=True, log_to_disk=True):
    if not is_training:
        metrics_file_path = metrics_dir + '/test_metrics.csv'
    else:
        metrics_file_path = metrics_dir + '/train_metrics.csv'

    metric = sklearn.metrics.f1_score(true_labels, predicted_labels)

    print('f1-score:  ' + str(metric))

    if log_to_disk:
        if os.path.exists(metrics_file_path):
            metrics_df = pd.read_csv(metrics_file_path)
        else:
            metrics_df = pd.DataFrame()

        metrics_df['f1-score'] = [metric]
        metrics_df.to_csv(metrics_file_path, index=False)

    return metric

def precision(predicted_labels=None, true_labels=None, metrics_dir='.', is_training=True, log_to_disk=True):
    if not is_training:
        metrics_file_path = metrics_dir + '/test_metrics.csv'
    else:
        metrics_file_path = metrics_dir + '/train_metrics.csv'

    metric = sklearn.metrics.precision_score(true_labels, predicted_labels)

    print('precision:  ' + str(metric))

    if log_to_disk:
        if os.path.exists(metrics_file_path):
            metrics_df = pd.read_csv(metrics_file_path)
        else:
            metrics_df = pd.DataFrame()

        metrics_df['precision'] = [metric]
        metrics_df.to_csv(metrics_file_path, index=False)

    return metric

def recall(predicted_labels=None, true_labels=None, metrics_dir='.', is_training=True, log_to_disk=True):
    if not is_training:
        metrics_file_path = metrics_dir + '/test_metrics.csv'
    else:
        metrics_file_path = metrics_dir + '/train_metrics.csv'

    metric = sklearn.metrics.recall_score(true_labels, predicted_labels)

    print('recall:  ' + str(metric))

    if log_to_disk:
        if os.path.exists(metrics_file_path):
            metrics_df = pd.read_csv(metrics_file_path)
        else:
            metrics_df = pd.DataFrame()

        metrics_df['recall'] = [metric]
        metrics_df.to_csv(metrics_file_path, index=False)

    return metric

def plot_feature_histograms(data, title='Feature Histogram', metrics_dir='.', file_name='histogram.png', figsize=(20, 20), bins=5):
    if type(data) is not pd.core.frame.DataFrame:
        if type(data) is np.ndarray:
            data = pd.DataFrame(data)

    hist = data.hist(figsize=figsize, bins=bins)
    plt.tight_layout()
    plt.savefig(f'{metrics_dir}/{file_name}')
    plt.close()

    if 'label' in data.columns:
        hist = data.loc[data.label == 0,:].hist(figsize=figsize, bins=bins)
        plt.savefig(f'{metrics_dir}/benign_{file_name}')
        plt.close()

        hist = data.loc[data.label == 1,:].hist(figsize=figsize, bins=bins)
        plt.savefig(f'{metrics_dir}/attack_{file_name}')
        plt.close()

def generate_confusion_matrix(predicted_labels, true_labels, normalize=None):
        cm = sklearn.metrics.confusion_matrix(true_labels, predicted_labels, normalize=normalize)
        return cm

def create_confusion_matrix_plot(ax_ref, cm, title='Confusion Matrix', target_names=None):
    # Generate plot of the confusion matrix
    ax_ref.tick_params(bottom=True, top=False, left=True, right=False)
    ax_ref.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    img = ax_ref.imshow(cm, cmap=plt.get_cmap('Blues'))
    plt.colorbar(img, ax=ax_ref)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax_ref.text(j, i, '{:0.4f}'.format(cm[i, j]),
             horizontalalignment='center',
             color='white' if cm[i, j] > thresh else 'black')

    ax_ref.set_xlabel('Predicted')
    ax_ref.set_ylabel('Truth')
    ax_ref.set_title(title)
    return ax_ref

def plot_confusion_matrix(predicted_labels=None, true_labels=None, metrics_dir='.', prefix='', is_training=True, title='Confusion Matrix', target_names=None):
    if not is_training:
        cm_file_path = metrics_dir + f'/{prefix}_test_cm.png'
        cm_norm_file_path = metrics_dir + f'/{prefix}_test_cm_norm.png'
    else:
        cm_file_path = metrics_dir + f'/{prefix}_train_cm.png'
        cm_norm_file_path = metrics_dir + f'/{prefix}_train_cm_norm.png'

    cm = generate_confusion_matrix(predicted_labels, true_labels, normalize=None)
    print(cm)
    fig = plt.figure()
    ax = fig.gca()

    ax = create_confusion_matrix_plot(ax, cm, title, target_names)
    plt.savefig(cm_file_path)
    plt.close()
    
    cm_norm = generate_confusion_matrix(predicted_labels, true_labels, normalize='all')
    print(cm_norm)
    fig = plt.figure()
    ax = fig.gca()

    ax = create_confusion_matrix_plot(ax, cm_norm, title, target_names)
    plt.savefig(cm_norm_file_path)
    plt.close()

    return cm, cm_norm

def plot_feature_importance(model, num_features, columns, metrics_dir='.', prefix=''):
    importances = model.model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    sorted_columns = columns[indices]

    print('Feature Ranking:')
    for f in range(num_features):
        print('%d. %s (%f)' % (f + 1, sorted_columns[f], importances[indices[f]]))

    plt.figure(figsize=[8,6])
    plt.title('Feature Importances')
    plt.bar(
        range(num_features),
        importances[indices],
        color='b',
        align='center'
    )

    plt.xticks(
        range(num_features), 
        sorted_columns,
        rotation='vertical',
        fontsize='xx-small',
    )
    plt.xlim([-1, num_features])
    plt.tight_layout()
    plt.savefig(f'{metrics_dir}/{prefix}_feature_importances.png')
    plt.close()
