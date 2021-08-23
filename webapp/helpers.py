import os
from config import SCENARIOS_FOLDER_PATH
import glob
import pickle
import pandas as pd
import json
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from config import TRAINING_DATA_FILE_NAME_REGEX, TEST_DATA_FILE_NAME_REGEX

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
    test_file = glob.glob(os.path.join(solution_set_path, TEST_DATA_FILE_NAME_REGEX))[0]
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
    train_file = glob.glob(os.path.join(solution_set_path, TRAINING_DATA_FILE_NAME_REGEX))[0]
    print("--- {}".format(glob.glob(os.path.join(solution_set_path,"train.*"))[0]))
    ext = os.path.splitext(train_file)[1]
    if ext == ".pkl":
        with open(train_file,'rb') as file:
            train = pickle.load(file)
    elif ext == ".csv":
        train = pd.read_csv(train_file)
    else:
        train = None
    
    return train

def compute_train_test_split(solution_set_path):
    training_dataset = read_train(solution_set_path)
    test_dataset = read_test(solution_set_path)
    print("TRAIN TEST SPLIT => TRAIN {0}, TEST {1}, SPLIT {2}".format(len(training_dataset), len(test_dataset), len(training_dataset)/len(test_dataset)))
    print("Lenght of test dataset {}".format(len(test_dataset)))
    print("Lenght of training dataset {}".format(len(train_dataset)))
    return len(training_dataset)/len(test_dataset)

def score_train_test_split():
    return 2.5

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
   
def create_info_modal(module_id, name, content, example):
    modal = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-info-circle"),
            id="{}_info_button".format(module_id), 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(name),
                dbc.ModalBody([content, html.Br(), html.Br(), dcc.Markdown(example) ]),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="{}_close".format(module_id), className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="{}_info_modal".format(module_id),
            is_open=False,
        ),
    ]
)
    return modal