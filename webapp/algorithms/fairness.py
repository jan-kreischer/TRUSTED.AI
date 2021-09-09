# -*- coding: utf-8 -*-
# functions for fairness score
from helpers import *

import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.algorithms.preprocessing import Reweighing

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
def analyse(model, training_dataset, test_dataset, factsheet, fairness_config):   
    output = dict(
        class_balance = class_balance_score(),
        statistical_parity_difference = statistical_parity_difference_score(model, training_dataset, test_dataset, factsheet),
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

# --- Class Balance ---
def class_balance_metric():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 

def class_balance_score():
    return result(score=np.random.randint(1,6), properties={"normalization_technique": "fairness"}) 


# --- Statistical Parity Difference ---
def statistical_parity_difference_score(model, training_dataset, test_dataset, factsheet):
    score = 0
    properties= {}
    favored_majority_ratio, favored_minority_ratio, statistical_parity_difference = statistical_parity_difference_metric(model, training_dataset, test_dataset, factsheet)
    print("statistical_parity_difference: {}".format(statistical_parity_difference))
    if abs(statistical_parity_difference) < 0.01:
        score = 5
    elif abs(statistical_parity_difference) < 0.025:
        score = 4
    elif abs(statistical_parity_difference) < 0.050:
        score = 3
    elif abs(statistical_parity_difference) < 0.075:
        score = 2
    elif abs(statistical_parity_difference) < 0.100:
        score = 1
    else:
        score = 0

    return result(score=score, properties=properties)

def statistical_parity_difference_metric(model, training_dataset, test_dataset, factsheet):
    target_column = ""
    if "general" in factsheet and "target_column" in factsheet["general"]:
        target_column = factsheet["general"]["target_column"]
        print("target_column: {}".format(target_column))

    favorable_outcome = ""
    if "fairness" in factsheet and "favorable_outcome" in factsheet["fairness"]:
        favorable_outcome = factsheet["fairness"]["favorable_outcome"]
        print("favorable_outcome: {}".format(favorable_outcome))
        favorable_outcome = eval(favorable_outcome, {"target_column": target_column}, {"target_column": target_column})

    protected_feature = ""
    if "fairness" in factsheet and "protected_feature" in factsheet["fairness"]:
        protected_feature = factsheet["fairness"]["protected_feature"]
        print("protected_feature: {}".format(protected_feature))

    protected = ""
    if "fairness" in factsheet and "protected_group" in factsheet["fairness"]:
        protected = factsheet["fairness"]["protected_group"]
        protected_feature = "Group"
        print("protected: {}".format(protected))

    protected = eval(protected, {"protected_feature": protected_feature}, {"protected_feature": protected_feature})

    #target_column = "Target"
    #favorable_outome = lambda x: x[target_column]==1

    #protected_feature = "Group"
    #protected = lambda x: x[protected_feature]==0
    protected_indices = training_dataset.apply(protected, axis=1)

    minority = training_dataset[protected_indices]
    minority_size = len(minority)
    favored_minority = minority[minority.apply(favorable_outcome, axis=1)]
    favored_minority_size = len(favored_minority)
    favored_minority_ratio = favored_minority_size/minority_size
    print("{0}/{1} = {2}".format(favored_minority_size, minority_size, favored_minority_ratio))

    majority = training_dataset[~protected_indices]
    majority_size = len(majority)
    favored_majority = majority[majority.apply(favorable_outcome, axis=1)]
    favored_majority_size = len(favored_majority)
    favored_majority_ratio = favored_majority_size/majority_size
    print("{0}/{1} = {2}".format(favored_majority_size, majority_size, favored_majority_ratio))
    return favored_majority_ratio, favored_minority_ratio, favored_minority_ratio - favored_majority_ratio


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

    
