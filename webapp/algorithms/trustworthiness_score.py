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
from algorithms.fairness_score import calc_fairness_score
from algorithms.explainability_score import calc_explainability_score
from algorithms.robustness_score import calc_robustness_score
from algorithms.methodology_score import calc_methodology_score
import collections
from helpers import read_solution

result = collections.namedtuple('result', 'score properties')

# define algo
def trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, methodology_config):
    output = dict(
        fairness       = calc_fairness_score(),
        explainability = calc_explainability_score(model, train_data, test_data, config_explainability, factsheet),
        robustness     = calc_robustness_score(model, train_data, test_data, config_robustness),
        methodology    = calc_methodology_score(model, train_data, test_data, factsheet, methodology_config)
    )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)

# calculate final score with weigths
def get_final_score(model, train_data, test_data, main_config, factsheet):
    config_fairness = main_config["fairness"]
    config_explainability = main_config["explainability"]
    config_robustness = main_config["robustness"]
    config_methodology = main_config["methodology"]
    
    result = trusting_AI_scores(model, train_data, test_data, factsheet, config_fairness, config_explainability, config_robustness, config_methodology)
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

def get_trust_score(final_score, config):
    if sum(config.values()) == 0:
        return 0
    return round(np.nansum(list(map(lambda x: final_score[x] * config[x], final_score.keys())))/np.sum(list(config.values())),1)

