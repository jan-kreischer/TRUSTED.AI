# -*- coding: utf-8 -*-
# functions for fairness score

import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.algorithms.preprocessing import Reweighing

import collections
result = collections.namedtuple('result', 'score properties')

# Load all necessary packages
#import sys
#sys.path.insert(1, "../")  

'''
from scipy.stats import chisquare

significance_level = 0.05
p_value = chisquare(absolute_class_occurences, ddof=0, axis=0).pvalue

print("P-Value of Chi Squared Test: {0}".format(p_value))
if p_value < significance_level:
    print("The data does not follow a unit distribution")
else:
    print("We can not reject the null hypothesis assuming that the data follows a unit distribution")
    
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 11);
xU, xL = x + 1, x - 1;
prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3);
prob = prob / prob.sum(); # normalize the probabilities so their sum is 1
sample = np.random.choice(x, size = 10000, p = prob);
(_, observed_occurences) = np.unique(sample, return_counts=True)
plt.hist(nums, bins = len(x))

p_value = chisquare(observed_occurences, ddof=0, axis=0).pvalue

print("P-Value of Chi Squared Test: {0}".format(p_value))

if p_value < significance_level:
    print("The data does not follow a unit distribution")
else:
    print("We can not reject the null hypothesis assuming that the data follows a unit distribution")
'''



# === Fairness Metrics ===
# --- Class Balance ---
def class_balance_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def class_balance_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Statistical Parity Difference ---
def statistical_parity_difference_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def statistical_parity_difference_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Equal Opportunity Difference ---
def equal_opportunity_difference_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def equal_opportunity_difference_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Average Odds Difference ---
def average_odds_difference_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def average_odds_difference_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Disparate Impact ---
def disparate_impact_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def disparate_impact_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Theil Index ---
def theil_index_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def theil_index_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Euclidean Distance ---
def euclidean_distance_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def euclidean_distance_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Mahalanobis Distance ---
def mahalanobis_distance_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def mahalanobis_distance_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Manhattan Distance ---
def manhattan_distance_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def manhattan_distance_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def calc_fairness_score():   
    output = dict(
        class_balance = class_balance_score(),
        statistical_parity_difference = statistical_parity_difference_score(),
        equal_opportunity_difference = equal_opportunity_difference_score(),
        average_odds_difference = average_odds_difference_score(),
        disparate_impact = disparate_impact_score(),
        theil_index = theil_index_score(),
        euclidean_distance = euclidean_distance_score(),
        mahalanobis_distance = mahalanobis_distance_score(),
        manhattan_distance = manhattan_distance_score()
    )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)
    
