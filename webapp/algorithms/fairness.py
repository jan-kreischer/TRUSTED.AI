# -*- coding: utf-8 -*-
# functions for fairness score
import numpy as np
np.random.seed(0)
from helpers import *
from scipy.stats import chisquare
import operator
import funcy

# === Fairness Metrics ===
def analyse(model, training_dataset, test_dataset, factsheet, fairness_config):   
    target_column = factsheet.get("general", {}).get("target_column", "")
    
    statistical_parity_difference_thresholds = fairness_config["score_statistical_parity_difference"]["thresholds"]["value"]
    overfitting_thresholds = fairness_config["score_overfitting"]["thresholds"]["value"]
    underfitting_thresholds = fairness_config["score_underfitting"]["thresholds"]["value"]
    equal_opportunity_difference_thresholds = fairness_config["score_equal_opportunity_difference"]["thresholds"]["value"]
    average_odds_difference_thresholds = fairness_config["score_average_odds_difference"]["thresholds"]["value"]
    
    output = dict(
        question_fairness = question_fairness_score(factsheet),
        class_balance = class_balance_score(training_dataset, target_column),
        underfitting = underfitting_score(model, training_dataset, test_dataset, factsheet, underfitting_thresholds),
        overfitting = overfitting_score(model, training_dataset, test_dataset, factsheet, overfitting_thresholds),
        statistical_parity_difference = statistical_parity_difference_score(model, training_dataset, test_dataset, factsheet, statistical_parity_difference_thresholds),
        equal_opportunity_difference = equal_opportunity_difference_score(model, test_dataset, factsheet, equal_opportunity_difference_thresholds),
        average_odds_difference = average_odds_difference_score(model, test_dataset, factsheet, average_odds_difference_thresholds)
    )
    
    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

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

    if p_value < significance_level:
        #The data does not follow a unit distribution
        return 0
    else:
        #We can not reject the null hypothesis assuming that the data follows a unit distribution"
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

# --- Underfitting ---
def underfitting_score(model, training_dataset, test_dataset, factsheet):
    try:
        properties = {}
        score = 0
        training_accuracy = compute_accuracy(model, training_dataset, factsheet, thresholds)
        test_accuracy = compute_accuracy(model, test_dataset, factsheet)
        if training_accuracy <= test_accuracy:
            # model could be underfitting.
            # for underfitting models the spread is negative
            accuracy_difference = training_accuracy - test_accuracy
            score = np.digitize(abs(accuracy_difference, thresholds, right=False)) + 1 
                
            if score == 5:
                properties["Conclusion"] = "Model is not underfitting"
            elif score == 4:
                properties["Conclusion"] = "Model mildly underfitting"
            elif score == 3:
                properties["Conclusion"] = "Model is slighly underfitting"
            elif score == 2:
                properties["Conclusion"] = "Model is underfitting"
            else:
                properties["Conclusion"] = "Model is strongly underfitting"
 
            properties["Training Accuracy"] = training_accuracy
            properties["Test Accuracy"] = test_accuracy
            properties["Train Test Accuracy Difference"] = training_accuracy - test_accuracy
    
            return result(score=score, properties=properties)
        else:
            return result(score=np.nan, properties={}) 
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={}) 

# --- Overfitting ---
def overfitting_score(model, training_dataset, test_dataset, factsheet, thresholds):
    try:
        properties = {"Conclusion": "Model is overfitting"}
        score = 0
        training_accuracy = compute_accuracy(model, training_dataset, factsheet)
        test_accuracy = compute_accuracy(model, test_dataset, factsheet)
        if training_accuracy >= test_accuracy:
            # model could be underfitting.
            # for underfitting models the spread is negative
            accuracy_difference = training_accuracy - test_accuracy
            score = np.digitize(abs(accuracy_difference, thresholds, right=False)) + 1 
            
            if score == 5:
                properties["Conclusion"] = "Model is not overfitting"
            elif score == 4:
                properties["Conclusion"] = "Model mildly overfitting"
            elif score == 3:
                properties["Conclusion"] = "Model is slighly overfitting"
            elif score == 2:
                properties["Conclusion"] = "Model is overfitting"
            else:
                properties["Conclusion"] = "Model is strongly overfitting"
                                
            properties["Training Accuracy"] = training_accuracy
            properties["Test Accuracy"] = test_accuracy
            properties["Train Test Accuracy Difference"] = training_accuracy - test_accuracy
    
            return result(score=score, properties=properties)
        else:
            return result(score=np.nan, properties={}) 
    except Exception as e:
        print(e)
        return result(score=np.nan, properties={}) 

def compute_accuracy(model, dataset, factsheet):
    try:
        target_column = factsheet.get("general", {}).get("target_column", {})
        X_data = dataset.drop(target_column, axis=1)
        y_data = dataset[target_column]

        y_true = y_data.values.flatten()
        if (isinstance(model, tf.keras.Sequential)):
            y_train_pred_proba = model.predict(X_train)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_data).flatten()
        return metrics.accuracy_score(y_true, y_pred)
    except Exception as e:
        print(e)
        raise

# --- Statistical Parity Difference ---
def statistical_parity_difference_score(model, training_dataset, test_dataset, factsheet, thresholds):
    #print("STATISTICAL PARITY TEST")
    try: 
        #print("Training data columns {}".format(training_dataset))
        score = 1
        favored_majority_ratio, favored_minority_ratio, statistical_parity_difference = statistical_parity_difference_metric(model, training_dataset, test_dataset, factsheet)
        properties = {"Favored Majority Ratio": favored_majority_ratio, "Favored Minority Ratio": favored_minority_ratio, "Statistical Parity Difference": statistical_parity_difference}
        score = np.digitize(statistical_parity_difference, thresholds, right=False) + 1 
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
        #print("BEFORE FAVORABLE_OUTCOME {}".format(favorable_outcome))
        favorable_outcome = eval(favorable_outcome, {"target_column": target_column}, {"target_column": target_column})
        #print("AFTER FAVORABLE_OUTCOME {}".format(favorable_outcome))
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
        #print("{0}/{1} = {2}".format(favored_minority_size, minority_size, favored_minority_ratio))

        majority = training_dataset[~protected_indices]
        majority_size = len(majority)
        favored_majority = majority[majority.apply(favorable_outcome, axis=1)]
        favored_majority_size = len(favored_majority)
        favored_majority_ratio = favored_majority_size/majority_size
        #print("{0}/{1} = {2}".format(favored_majority_size, majority_size, favored_majority_ratio))
        return favored_majority_ratio, favored_minority_ratio, favored_minority_ratio - favored_majority_ratio
    except Exception as e:
        print(e)
        raise


# --- Equal Opportunity Difference ---
def equal_opportunity_difference_score(model, test_dataset, factsheet, thresholds):
    try:
        properties = {}
        score=np.nan
        properties["Metric Description"] = "Difference in true positive rates between protected and unprotected group."
        tpr_protected, tpr_unprotected = true_positive_rates(model, test_dataset, factsheet)
        properties["TPR Unprotected Group"] = "P(y_hat=favorable|y_true=favorable, protected=False) = {}".format(tpr_unprotected)
        properties["TPR Protected Group"] = "P(y_hat=favorable|y_true=favorable, protected=True) = {}".format(tpr_protected)  
        equal_opportunity_difference = tpr_protected - tpr_unprotected
        properties["Equal Opportunity Difference"] = "TPR Protected Group - TPR Unprotected Group = {}".format(equal_opportunity_difference)
        
        score = np.digitize(abs(equal_opportunity_difference, thresholds, right=False)) + 1 
        
        return result(score=score, properties=properties) 
    except Exception as e:
        print("ERROR in equal_opportunity_difference_score: {}".format(e))
        return result(score=np.nan, properties={})
        
# --- Average Odds Difference ---
def average_odds_difference_score(model, test_dataset, factsheet, thresholds):
    try:
        score = np.nan
        properties = {}
        properties["Metric Description"] = "Is the average of difference in false positive rates and true positive rates between the protected and unprotected group"
        fpr_protected, fpr_unprotected = false_positive_rates(model, test_dataset, factsheet)
        properties["FPR Unprotected Group"] = "P(y_hat=favorable|y_true=unfavorable, protected=False) = {}".format(fpr_unprotected)
        properties["FPR Protected Group"] = "P(y_hat=favorable|y_true=unfavorable, protected=True) = {}".format(fpr_protected) 
        
        tpr_protected, tpr_unprotected = true_positive_rates(model, test_dataset, factsheet)
        properties["TPR Unprotected Group"] = "P(y_hat=favorable|y_true=favorable, protected=False) = {}".format(tpr_unprotected)
        properties["TPR Protected Group"] = "P(y_hat=favorable|y_true=favorable, protected=True) = {}".format(tpr_protected) 
        
        average_odds_difference = ((tpr_protected - tpr_unprotected) + (fpr_protected - fpr_unprotected))/2
        properties["Average Odds Difference"] = "0.5*(TPR Protected - TPR Unprotected) + 0.5*(FPR Protected - FPR Unprotected): {}".format(average_odds_difference)
    
        score = np.digitize(abs(average_odds_difference, thresholds, right=False)) + 1 
            
        return result(score=score, properties=properties) 
    except Exception as e:
        print("ERROR in average_odds_difference_score(): {}".format(e))
        raise
        
def false_positive_rates(model, test_dataset, factsheet):
    try: 
        target_column = factsheet.get("general", {}).get("target_column", "")
        data = test_dataset.copy(deep=True)
        
        X_data = data.drop(target_column, axis=1)
        if (isinstance(model, tf.keras.Sequential)):
            y_pred_proba = model.predict(X_data)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_data).flatten()
        data['y_pred'] = y_pred.tolist()

        favorable_outcome_definition = factsheet.get("fairness", {}).get("favorable_outcome", None)
        favorable_prediction = eval(favorable_outcome_definition, {"target_column": 'y_pred'}, {"target_column": 'y_pred'})
        favorable_outcome = eval(favorable_outcome_definition, {"target_column": target_column}, {"target_column": target_column})
        unfavorable_outcome = funcy.compose(operator.not_, favorable_outcome)
        
        protected_feature = factsheet.get("fairness", {}).get("protected_feature", None)
        protected_group_definition = factsheet.get("fairness", {}).get("protected_group", None)
        protected = eval(protected_group_definition, {"protected_feature": protected_feature}, {"protected_feature": protected_feature})
        
        #1. Divide into protected and unprotected group
        protected_indices = data.apply(protected, axis=1)
        protected_group = data[protected_indices]
        unprotected_group = data[~protected_indices]

        #2. Compute the number of negative samples y_true=False for the protected and unprotected group.
        protected_group_true_unfavorable = protected_group[protected_group.apply(unfavorable_outcome, axis=1)]
        unprotected_group_true_unfavorable = unprotected_group[unprotected_group.apply(unfavorable_outcome, axis=1)]
        protected_group_n_true_unfavorable = len(protected_group_true_unfavorable)
        unprotected_group_n_true_unfavorable = len(unprotected_group_true_unfavorable)
        print("protected_group_n_true_unfavorable {}".format(protected_group_n_true_unfavorable))
        print("unprotected_group_true_unfavorable {}".format(unprotected_group_true_unfavorable))

        #3. Calculate the number of false positives for the protected and unprotected group
        protected_group_true_unfavorable_pred_favorable = protected_group_true_unfavorable[protected_group_true_unfavorable.apply(favorable_prediction, axis=1)]
        unprotected_group_true_unfavorable_pred_favorable = unprotected_group_true_unfavorable[unprotected_group_true_unfavorable.apply(favorable_prediction, axis=1)]
        protected_group_n_true_unfavorable_pred_favorable = len(protected_group_true_unfavorable_pred_favorable)
        unprotected_group_n_true_unfavorable_pred_favorable = len(unprotected_group_true_unfavorable_pred_favorable)

        #4. Calculate fpr for both groups.
        fpr_protected = protected_group_n_true_unfavorable_pred_favorable/protected_group_n_true_unfavorable
        fpr_unprotected = unprotected_group_n_true_unfavorable_pred_favorable/unprotected_group_n_true_unfavorable
        print("protected_fpr: {}".format(fpr_protected))
        print("unprotected_fpr: {}".format(fpr_unprotected))
        return fpr_protected, fpr_unprotected

    except Exception as e:
        print("ERROR in false_positive_rates(): {}".format(e))
        raise
        
def true_positive_rates(model, test_dataset, factsheet):
    try: 
        target_column = factsheet.get("general", {}).get("target_column", "")
        data = test_dataset.copy(deep=True)
        
        X_data = data.drop(target_column, axis=1)
        if (isinstance(model, tf.keras.Sequential)):
            y_pred_proba = model.predict(X_data)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_data).flatten()
        data['y_pred'] = y_pred.tolist()

        favorable_outcome_definition = factsheet.get("fairness", {}).get("favorable_outcome", None)
        favorable_prediction = eval(favorable_outcome_definition, {"target_column": 'y_pred'}, {"target_column": 'y_pred'})
        favorable_outcome = eval(favorable_outcome_definition, {"target_column": target_column}, {"target_column": target_column})
        
        protected_feature = factsheet.get("fairness", {}).get("protected_feature", None)
        protected_group_definition = factsheet.get("fairness", {}).get("protected_group", None)
        protected = eval(protected_group_definition, {"protected_feature": protected_feature}, {"protected_feature": protected_feature})
        
        favored_indices = data.apply(favorable_outcome, axis=1)
        protected_indices = data.apply(protected, axis=1)

        favored_samples = data[favored_indices]
        protected_favored_samples = favored_samples[protected_indices]
        unprotected_favored_samples = favored_samples[~protected_indices]

        num_unprotected_favored_true = len(unprotected_favored_samples)
        num_unprotected_favored_pred = len(unprotected_favored_samples[unprotected_favored_samples.apply(favorable_prediction, axis=1)])
        tpr_unprotected = num_unprotected_favored_pred/num_unprotected_favored_true

        num_protected_favored_true = len(protected_favored_samples)
        num_protected_favored_pred = len(protected_favored_samples[protected_favored_samples.apply(favorable_prediction, axis=1)])
        tpr_protected = num_protected_favored_pred / num_protected_favored_true 
        return tpr_protected, tpr_unprotected

    except Exception as e:
        print("ERROR in equal_opportunity_difference_metric: {}".format(e))
        raise
    
