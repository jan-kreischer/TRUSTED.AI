# -*- coding: utf-8 -*-
import numpy as np
import collections

result = collections.namedtuple('result', 'score properties')


# === Methodology Metrics ===
def score_Normalization():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Test_F1_score():
    return result(score=np.random.randint(1,6), properties={}) 

# --- Train-Test-Split ---
def is_between(a, x, b):
    return min(a, b) < x < max(a, b)

def metric_train_test_split_ratio(training_dataset, test_dataset):
    n_train = len(training_dataset)
    n_test = len(test_dataset)
    n = n_train + n_test
    return round(n_train/n*100), round(n_test/n*100)
    
def metric_train_test_split(training_dataset, test_dataset):

    print("TRAIN TEST SPLIT => TRAIN {0}, TEST {1}, SPLIT {2}".format(len(training_dataset), len(test_dataset), len(training_dataset)/len(test_dataset)))
    return len(training_dataset)/len(test_dataset)

def score_train_test_split(training_dataset, test_dataset):
    train_test_split = metric_train_test_split(training_dataset, test_dataset)
    score = 0
    training_data_ratio, test_data_ratio = metric_train_test_split_ratio(training_dataset, test_dataset)
    properties= {"training_data_ratio": training_data_ratio, "test_data_ratio": test_data_ratio}
    print("percentage_train {0}, percentage_test {1}".format(training_data_ratio, test_data_ratio))
    
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

# --- Regularization ---
def score_Regularization():
    return result(score=np.random.randint(1,6), properties={}) 

def score_Test_Accuracy():
    return result(score=np.random.randint(1,6), properties={}) 

def calc_methodology_score(model, training_data, test_data, factsheet, methodology_config):
    output = dict(
        Normalization    = score_Normalization(),
        Treatment_of_Corrupt_Values     = score_Test_F1_score(),
        Train_Test_Split = score_train_test_split(training_data, test_data),
        Regularization   = score_Regularization(),
        Feature_Filtering    = score_Test_Accuracy(),
        Treatment_of_Categorical_Features  = score_Test_Accuracy()
                 )
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)