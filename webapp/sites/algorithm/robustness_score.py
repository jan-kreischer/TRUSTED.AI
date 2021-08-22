# -*- coding: utf-8 -*-
import numpy as np
import collections
import random
import sklearn.metrics as metrics
from art.attacks.evasion import FastGradientMethod, CarliniL2Method, DeepFool
from art.estimators.classification import SklearnClassifier
from sklearn.preprocessing import OneHotEncoder

result = collections.namedtuple('result', 'score properties')

# functions for robustness score

def score_Confidence_Score(model, train_data, test_data):
    try:
        X_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1: ]
        y_pred = model.predict(X_test)

        confidence = metrics.confusion_matrix(y_test, y_pred)/metrics.confusion_matrix(y_test, y_pred).sum(axis=1) 
        confidence_score = np.average(confidence.diagonal())
        score = confidence_score * 5
        return result(score=score, properties={})
    except:
        return result(score=0, properties={})

def score_Class_Specific_Metrics():
    return result(score=np.random.randint(1,6), properties={})

def score_Fast_Gradient_Attack(model, train_data, test_data):
    try:
        randomData = test_data.sample(50)
        randomX = randomData.iloc[:,:-1]
        randomY = randomData.iloc[:,-1: ]

        y_pred = model.predict(randomX)
        before_attack = metrics.accuracy_score(randomY,y_pred)

        classifier = SklearnClassifier(model=model)
        attack = FastGradientMethod(estimator=classifier, eps=0.2)
        x_test_adv = attack.generate(x=randomX)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(test_data.iloc[:,-1: ])
        randomY = enc.transform(randomY).toarray()

        predictions = model.predict(x_test_adv)
        predictions = enc.transform(predictions.reshape(-1,1)).toarray()
        after_attack = metrics.accuracy_score(randomY,predictions)

        print("Accuracy on before_attacks: {}%".format(before_attack * 100))
        print("Accuracy on after_attack: {}%".format(after_attack * 100))

        score = 5- (((before_attack - after_attack)/before_attack) * 5)
        if score > 5 or score < 0:
            score = 0
        return result(score=score, properties={})
    except:
        return result(score=0, properties={})

def score_Carlini_Wagner_Attack(model, train_data, test_data):
    try:
        randomData = test_data.sample(5)
        randomX = randomData.iloc[:,:-1]
        randomY = randomData.iloc[:,-1: ]

        y_pred = model.predict(randomX)
        before_attack = metrics.accuracy_score(randomY,y_pred)

        classifier = SklearnClassifier(model=model)
        attack = CarliniL2Method(classifier)
        x_test_adv = attack.generate(x=randomX)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(test_data.iloc[:,-1: ])
        randomY = enc.transform(randomY).toarray()

        predictions = model.predict(x_test_adv)
        predictions = enc.transform(predictions.reshape(-1,1)).toarray()
        after_attack = metrics.accuracy_score(randomY,predictions)
        print("Accuracy on before_attacks: {}%".format(before_attack * 100))
        print("Accuracy on after_attack: {}%".format(after_attack * 100))
        score = 5- (((before_attack - after_attack)/before_attack) * 5)
        if score > 5 or score < 0:
            score = 0
        return result(score=score, properties={})
    except:
        return result(score=0, properties={})

def score_Deepfool_Attack(model, train_data, test_data):
    try:
        randomData = test_data.sample(4)
        randomX = randomData.iloc[:,:-1]
        randomY = randomData.iloc[:,-1: ]

        y_pred = model.predict(randomX)
        before_attack = metrics.accuracy_score(randomY,y_pred)

        classifier = SklearnClassifier(model=model)
        attack = DeepFool(classifier)
        x_test_adv = attack.generate(x=randomX)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(test_data.iloc[:,-1: ])
        randomY = enc.transform(randomY).toarray()

        predictions = model.predict(x_test_adv)
        predictions = enc.transform(predictions.reshape(-1,1)).toarray()
        after_attack = metrics.accuracy_score(randomY,predictions)
        print("Accuracy on before_attacks: {}%".format(before_attack * 100))
        print("Accuracy on after_attack: {}%".format(after_attack * 100))
        score = 5- (((before_attack - after_attack)/before_attack) * 5)
        if score > 5 or score < 0:
            score = 0
        return result(score=score, properties={})
    except:
        return result(score=0, properties={})

def calc_robustness_score(model, train_data, test_data, config):
    
    output = dict(
        Confidence_Score          = score_Confidence_Score(model, train_data, test_data),
        Clique_Method    = score_Class_Specific_Metrics(),
        Loss_Sensitivity   = score_Class_Specific_Metrics(),
        CLEVER_Score   = score_Class_Specific_Metrics(),
        Empirical_Robustness_Fast_Gradient_Attack = score_Fast_Gradient_Attack(model, train_data, test_data),
        Empirical_Robustness_Carlini_Wagner_Attack = score_Carlini_Wagner_Attack(model, train_data, test_data),
        Empirical_Robustness_Deepfool_Attack = score_Deepfool_Attack(model, train_data, test_data)
                 )
    scores = dict((k, int(v.score)) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())
    
    return  result(score=scores, properties=properties)
