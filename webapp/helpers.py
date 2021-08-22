import os
from config import SCENARIOS_FOLDER_PATH
import glob
import pickle
import pandas as pd
import json

def get_solution_sets():
    problem_sets = [(f.name, f.path) for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir() and not f.name.startswith('.')]
    options = []
    for problem_set_name, problem_set_path in problem_sets:
        solution_sets = [(f.name, f.path) for f in os.scandir(problem_set_path) if f.is_dir() and not f.name.startswith('.')]
        for solution_set_name, solution_set_path in solution_sets:
            options.append({"label": problem_set_name + " > " + solution_set_name, "value": solution_set_path})
    return options

def list_of_scenarios():
    problem_sets = [(f.name, f.path) for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir() and not f.name.startswith('.')]
    options = []
    for problem_set_name, problem_set_path in problem_sets:
        options.append({"label": problem_set_name, "value": problem_set_path})
    return options

def read_test(solution_set_path):
    
    #test data
    test_file = glob.glob(os.path.join(solution_set_path,"test.*"))[0]
    ext = os.path.splitext(test_file)[1]
    if ext == ".pkl":
        with open(test_file,'rb') as file:
            test = pickle.load(file)
    elif ext == ".csv":
        test = pd.read_csv(test_file)
    else:
        test = None
    
    return test
        
def read_train(solution_set_path):
    
 #train data
    train_file = glob.glob(os.path.join(solution_set_path,"test.*"))[0]
    ext = os.path.splitext(train_file)[1]
    if ext == ".pkl":
        with open(train_file,'rb') as file:
            train = pickle.load(file)
    elif ext == ".csv":
        train = pd.read_csv(train_file)
    else:
        train = None
    
    return train

def read_model(solution_set_path):
 
    #model
    with open(os.path.join(solution_set_path, "model.sav"),'rb') as file:
        model = pickle.load(file)
    
    return model

def read_factsheet(solution_set_path):
 
    #factsheet
    with open(os.path.join(solution_set_path, "factsheet.json"),'rb') as f:
                factsheet = json.loads(f.read())
    return factsheet

def read_scenario(solution_set_path):
    
    test = read_test(solution_set_path)
    train = read_train(solution_set_path)
    model = read_model(solution_set_path)
    factsheet = read_factsheet(solution_set_path)
                
    return test, train, model, factsheet
    