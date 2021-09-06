# -*- coding: utf-8 -*-
import numpy as np
import collections
from helpers import *


result = collections.namedtuple('result', 'score properties')


# === Methodology Metrics ===
# --- Normalization ---
def normalization_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    normalization_technique = normalization_metric(factsheet)
    score = 0
    properties= {"normalization_technique": normalization_technique}
    #properties["html"] = html.H3("Normalization Technique: {}".format(normalization_technique))
    if normalization_technique == "none":
        score = 0
    elif normalization_technique == "not specified":
        score = 1
    elif normalization_technique == "normalization":
        score = 4
    elif normalization_technique == "standardization":
        score = 5
    return result(score=score, properties=properties)

def normalization_metric(factsheet):
    if "methodology" in factsheet and "data_normalization" in factsheet["methodology"]:
        return factsheet["methodology"]["data_normalization"]
    else:
        return "not specified"
    
# --- f1 ---

def f1_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    return result(score=np.random.randint(1,6), properties={}) 

# --- Train-Test-Split ---
def train_test_split_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    score = 0
    training_data_ratio, test_data_ratio = train_test_split_metric(training_dataset, test_dataset)
    properties= {"training_data_ratio": training_data_ratio, "test_data_ratio": test_data_ratio}
    
    if is_between(79, training_data_ratio, 81):
        score = 5
    elif is_between(85, training_data_ratio, 75):
        score = 4
    elif is_between(90, training_data_ratio, 70):
        score = 3
    elif is_between(95, training_data_ratio, 60):
        score = 2
    elif is_between(97.5, training_data_ratio, 50):
        score = 1
    else:
        score = 0 
    return result(score=score, properties=properties)

def train_test_split_metric(training_dataset, test_dataset):
    n_train = len(training_dataset)
    n_test = len(test_dataset)
    n = n_train + n_test
    return round(n_train/n*100), round(n_test/n*100)

def is_between(a, x, b):
    return min(a, b) < x < max(a, b)

# --- Regularization ---
def regularization_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    score = 0
    regularization = regularization_metric(factsheet)
    properties= {"regularization_technique": regularization}
    if regularization == "elasticnet_regression":
        score = 5
    elif regularization == "lasso_regression" or regularization == "lasso_regression":
        score = 4
    elif regularization == NOT_SPECIFIED:
        score = 1
    else:
        score = 0
    return result(score=score, properties=properties)

def regularization_metric(factsheet):
    if "methodology" in factsheet and "regularization" in factsheet["methodology"]:
        return factsheet["methodology"]["regularization"]
    else:
        return NOT_SPECIFIED
    

    
# --- Factsheet Completeness ---

def factsheet_completeness_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    score = 0
    properties= dict()
    n = len(GENERAL_INPUTS)
    ctr = 0
    for e in GENERAL_INPUTS:
        if "general" in factsheet and e in factsheet["general"]:
            ctr+=1
            properties[e] = "present"
        else:
            properties[e] = "missing"
    score = round(ctr/n*5)
    return result(score=score, properties=properties)
            
    return result(score=np.random.randint(1,6), properties={}) 

def factsheet_completeness_metric(factsheet):
    print("hi")




# --- Test Accuracy ---

def test_accuracy_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    return result(score=np.random.randint(1,6), properties={}) 

def calc_methodology_score(model, training_dataset, test_dataset, factsheet, methodology_config):
    metrics = list_of_metrics("methodology")
    print(metrics)
    output = dict(
    #for metric in metrics:
        #output[metric] = exec("%s_score(model, training_dataset, test_dataset, factsheet, methodology_config)" % metric)
        normalization  = normalization_score(model, training_dataset, test_dataset, factsheet, methodology_config),
        regularization   = regularization_score(model, training_dataset, test_dataset, factsheet, methodology_config),
        train_test_split = train_test_split_score(model, training_dataset, test_dataset, factsheet, methodology_config),
        factsheet_completeness= factsheet_completeness_score(model, training_dataset, test_dataset, factsheet, methodology_config)
    )
    
    for metric in metrics:
        exec("{0} = {0}_score(model, training_dataset, test_dataset, factsheet, methodology_config)".format(metric))
        print("METRIC {}".format(metric))
        exec("print({0})".format(metric))

        #treatment_of_corrupt_values     = f1_score(model, training_dataset, test_dataset, factsheet, methodology_config),
        #feature_filtering    = test_accuracy_score(model, training_dataset, test_dataset, factsheet, methodology_config),
        #treatment_of_categorical_features  = test_accuracy_score(model, training_dataset, test_dataset, factsheet, methodology_config)
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)