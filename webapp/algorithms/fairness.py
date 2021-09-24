# -*- coding: utf-8 -*-
# functions for fairness score
from helpers import *
from scipy.stats import chisquare



import numpy as np
np.random.seed(0)

#from aif360.datasets import GermanDataset
#from aif360.metrics import BinaryLabelDatasetMetric
#from aif360.datasets import BinaryLabelDataset, StandardDataset
#from aif360.algorithms.preprocessing import Reweighing

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
    target_column = factsheet.get("general", {}).get("target_column", "")
    
    output = dict(
        question_fairness = question_fairness_score(factsheet),
        class_balance = class_balance_score(training_dataset, target_column),
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
    
    print("RESULT PROPERTIES {}".format(properties))
    return  result(score=scores, properties=properties)

# --- Question Fairness ---
def question_fairness_score(factsheet):
    try:
        score = factsheet.get("fairness", {}).get("question_fairness", np.nan)
        properties = {"Question Fairness": "{}".format(score)}
        return result(score=score, properties=properties) 
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={}) 
    

# --- Class Balance ---
def class_balance_metric(training_data, target_column):
    absolute_class_occurences = training_data[target_column].value_counts().sort_index().to_numpy()
    significance_level = 0.05
    p_value = chisquare(absolute_class_occurences, ddof=0, axis=0).pvalue

    print("P-Value of Chi Squared Test: {0}".format(p_value))
    if p_value < significance_level:
        print("The data does not follow a unit distribution")
        return 0
    else:
        print("We can not reject the null hypothesis assuming that the data follows a unit distribution")
        return 1


def class_balance_score(training_data, target_column):
    try:
        class_balance = class_balance_metric(training_data, target_column)
        properties = {}
        if(class_balance == 1):
            score = 5
        else:
            score = 1
        properties["class balance score"] = score
        return result(score=score, properties=properties)     
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={}) 

# --- Statistical Parity Difference ---
def statistical_parity_difference_score(model, training_dataset, test_dataset, factsheet):
    print("STATISTICAL PARITY TEST")
    try: 
        print("Training data columns {}".format(training_dataset))
        score = 1
        favored_majority_ratio, favored_minority_ratio, statistical_parity_difference = statistical_parity_difference_metric(model, training_dataset, test_dataset, factsheet)
        properties = {"Favored Majority Ratio": favored_majority_ratio, "Favored Minority Ratio": favored_minority_ratio, "Statistical Parity Difference": statistical_parity_difference}
        print("statistical_parity_difference: {}".format(statistical_parity_difference))
        if abs(statistical_parity_difference) < 0.01:
            score = 5
        elif abs(statistical_parity_difference) < 0.025:
            score = 4
        elif abs(statistical_parity_difference) < 0.050:
            score = 3
        elif abs(statistical_parity_difference) < 0.075:
            score = 2
        else:
            score = 1
        return result(score=score, properties=properties)
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={})

def statistical_parity_difference_metric(model, training_dataset, test_dataset, factsheet):
    try: 
        protected_feature = factsheet.get("fairness", {}).get("protected_feature", None)
        protected = factsheet.get("fairness", {}).get("protected_group", None)
        target_column = factsheet.get("general", {}).get("target_column", None)
        favorable_outcome = factsheet.get("fairness", {}).get("favorable_outcome", None)

        protected = eval(protected, {"protected_feature": protected_feature}, {"protected_feature": protected_feature})
        print("BEFORE FAVORABLE_OUTCOME {}".format(favorable_outcome))
        favorable_outcome = eval(favorable_outcome, {"target_column": target_column}, {"target_column": target_column})
        print("AFTER FAVORABLE_OUTCOME {}".format(favorable_outcome))
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
    except Exception as e:
        print(e)
        return 0, 0, 0


# --- Equal Opportunity Difference ---
def equal_opportunity_difference_score():
    return result(score=np.nan, properties={}) 

def equal_opportunity_difference_metric():
    return result(score=np.nanp, properties={}) 


# --- Average Odds Difference ---
def average_odds_difference_score():
    return result(score=np.nan, properties={}) 

def average_odds_difference_metric():
    return result(score=np.nan, properties={}) 


# --- Disparate Impact ---
def disparate_impact_score():
    return result(score=np.nan, properties={}) 

def disparate_impact_metric():
    return result(score=np.nan, properties={}) 


# --- Theil Index ---
def theil_index_score():
    return result(score=np.nan, properties={}) 

def theil_index_metric():
    return result(score=np.nan, properties={}) 


# --- Euclidean Distance ---
def euclidean_distance_score():
    return result(score=np.nan, properties={}) 

def euclidean_distance_metric():
    return result(score=np.nan, properties={}) 


# --- Mahalanobis Distance ---
def mahalanobis_distance_score():
    return result(score=np.nan, properties={}) 

def mahalanobis_distance_metric():
    return result(score=np.nan, properties={}) 


# --- Manhattan Distance ---
def manhattan_distance_score():
    return result(score=np.nan, properties={}) 

def manhattan_distance_metric():
    return result(score=np.nan, properties={}) 

    
