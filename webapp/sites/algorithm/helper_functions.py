# -*- coding: utf-8 -*-
import os
import time
import random
import numpy as np
import pickle
import statistics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json
from config import *
from math import pi
import sklearn.metrics as metrics
from sites.algorithm.fairness_score import calc_fairness_score
from sites.algorithm.explainability_score import calc_explainability_score
from sites.algorithm.robustness_score import calc_robustness_score
from sites.algorithm.methodology_score import calc_methodology_score
import collections
from helpers import read_solution

result = collections.namedtuple('result', 'score properties')

## get inputs
def get_case_inputs(case):
    scenario_path = os.path.join("Validation", case)

    # with open(os.path.join(scenario_path, "factsheet.txt")) as file:
    #     factsheet = json.load(file)
        
    with open(os.path.join(scenario_path, "model.sav"),'rb') as file:
        model = pickle.load(file)

    with open(os.path.join(scenario_path, "train_data.pkl"),'rb') as file:
        train_data = pickle.load(file)

    with open(os.path.join(scenario_path, "test_data.pkl"),'rb') as file:
        test_data = pickle.load(file)

    return [model, train_data, test_data]

def draw_spider(categories, values, ax, color='lightblue', title='Trusting AI Final Score'):
    
    values += values[:1]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.xaxis.set_tick_params(pad=6)
    # Add a title
    plt.title(title, size=11, y=1.1,
             bbox=dict(facecolor=color, edgecolor=color, pad=2.0))

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1,2,3], ["1","2","3"], color="grey", size=7)
    plt.ylim(0,4)

    # Plot data
    ax.plot(angles, values,color=color, linewidth=1, linestyle='solid')

    # Fill area
    ax.fill(angles, values, color=color, alpha=0.4)
    
def draw_bar_plot(categories, values, ax, color='lightblue', title='Trusting AI Final Score',size=12):
    
    # drop top and right spine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.grid(True,axis='y',zorder=0)
    
    # create barplot
    x_pos = np.arange(len(categories))
    plt.bar(x_pos, values,zorder=4,color=color)

    for i, v in enumerate(values):
        plt.text(i , v+0.1 , str(v),color='dimgray', fontweight='bold',ha='center')

    # Create names on the x-axis
    plt.xticks(x_pos, categories,size=size,wrap=True)

    plt.yticks([1,2,3,4], ["1","2","3","4"], size=12)
    plt.ylim(0,4)
    
    if isinstance(color, list):
        plt.title(title, size=11, y=1.1)
    else:
        plt.title(title, size=11, y=1.1,
             bbox=dict(facecolor=color, edgecolor=color, pad=2.0))
        plt.title(title, size=11, y=1.1)
        
# define algo
def trusting_AI_scores(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology):
    output = dict(
        fairness       = calc_fairness_score(),
        explainability = calc_explainability_score(model, train_data, test_data, config_explainability),
        robustness     = calc_robustness_score(model, train_data, test_data, config_robustness),
        methodology    = calc_methodology_score()
    )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)

# calculate final score with weigths
def get_final_score(model, train_data, test_data, main_config):
    config_fairness = main_config["fairness"]
    config_explainability = main_config["explainability"]
    config_robustness = main_config["robustness"]
    config_methodology = main_config["methodology"]
    
    result = trusting_AI_scores(model, train_data, test_data, config_fairness, config_explainability, config_robustness, config_methodology)
    scores = result.score
    properties = result.properties
    final_scores = dict()
    for pillar, item in scores.items():
        config = eval("config_"+pillar)
        weighted_scores = list(map(lambda x: scores[pillar][x] * config["weights"][x], scores[pillar].keys()))
        sum_weights = np.nansum(np.array(list(config["weights"].values()))[~np.isnan(weighted_scores)])
        if sum_weights == 0:
            result = 0
        else:
            result = round(np.nansum(weighted_scores)/sum_weights,1)
        final_scores[pillar] = result

    return final_scores, scores, properties

#config = {'fairness': 0.25, 'explainability': 0.25, 'robustness': 0.25, 'methodology': 0.25}

def get_trust_score(final_score, config):
    if sum(config.values()) == 0:
        return 0
    return round(np.nansum(list(map(lambda x: final_score[x] * config[x], final_score.keys())))/np.sum(list(config.values())),1)
    
def get_performance_metrics(model, test_data, target_column):
    
    test_data = test_data.copy()
    
    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
    else:
        X_test = test_data.iloc[:,:DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_data.iloc[:,DEFAULT_TARGET_COLUMN_INDEX: ]
    
    y_true =  y_test.values.flatten()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    labels = np.unique(np.array([y_pred,y_true]).flatten())

    performance_metrics = pd.DataFrame({
        "accuracy" :  [metrics.accuracy_score(y_true, y_pred)],
        #"global recall" :  [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted recall" : [metrics.recall_score(y_true, y_pred,average="weighted")],
        #"global precision" : [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted precision" : [metrics.precision_score(y_true, y_pred,average="weighted")],
        #"global f1 score" :  [metrics.f1_score(y_true, y_pred,average="micro")],
        "class weighted f1 score" :  [metrics.f1_score(y_true, y_pred,average="weighted")],
        "cross-entropy loss" : [metrics.log_loss(y_true, y_pred_proba)],
        "ROC AUC" : [metrics.roc_auc_score(y_true, y_pred_proba,average="weighted", multi_class='ovr')]
    }).round(decimals=2)
    return performance_metrics