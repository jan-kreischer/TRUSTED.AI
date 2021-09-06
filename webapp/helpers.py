import os
from config import SCENARIOS_FOLDER_PATH
import glob
import pickle
import pandas as pd
import json
import base64
import io
import numpy as np
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from config import *
from tensorflow.keras.models import load_model
import time
import random
import pickle
import seaborn as sn
import matplotlib.pyplot as plt
from math import pi
import sklearn.metrics as metrics
import collections
from helpers import *

def get_performance_metrics(model, test_data, target_column):
    
    test_data = test_data.copy()
    
    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
    else:
        X_test = test_data.iloc[:,:DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_data.iloc[:,DEFAULT_TARGET_COLUMN_INDEX: ]
    
    y_true =  y_test.values.flatten()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    labels = np.unique(np.array([y_pred,y_true]).flatten())

    performance_metrics = pd.DataFrame({
        "accuracy" :  [metrics.accuracy_score(y_true, y_pred)],
        #"global recall" :  [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted recall" : [metrics.recall_score(y_true, y_pred,average="weighted")],
        #"global precision" : [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted precision" : [metrics.precision_score(y_true, y_pred,average="weighted")],
        #"global f1 score" :  [metrics.f1_score(y_true, y_pred,average="micro")],
        "class weighted f1 score" :  [metrics.f1_score(y_true, y_pred,average="weighted")],
        "cross-entropy loss" : [metrics.log_loss(y_true, y_pred_proba)],
        "ROC AUC" : [metrics.roc_auc_score(y_true, y_pred_proba,average="weighted", multi_class='ovr')]
    }).round(decimals=2)
    return performance_metrics

def show_star_rating(rating):
    stars = []
    for i in range(0,5):
        if i+0.99 <= rating:
            stars.append(html.I(className="fas fa-star"))
        elif i+0.49 < rating:
            stars.append(html.I(className="fas fa-star-half-alt"))
        else:
            stars.append(html.I(className="far fa-star"))
    return stars

def get_solution_sets():
    problem_sets = [(f.name, f.path) for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir() and not f.name.startswith('.')]
    options = []
    for problem_set_name, problem_set_path in problem_sets:
        solution_sets = [(f.name, f.path) for f in os.scandir(os.path.join(problem_set_path, SOLUTIONS_FOLDER)) if f.is_dir() and not f.name.startswith('.')]
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
    if solution_set_path is None:
        return
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

# Load .joblib or .pickle model
def read_model(solution_set_path):
    model_file = glob.glob(os.path.join(solution_set_path, MODEL_REGEX))[0]
    print("model_file: {}".format(model_file))
    file_extension = os.path.splitext(model_file)[1]
    print("file extension of model to load {0}".format(file_extension))
    pickle_file_extensions = [".sav", ".pkl", ".pickle"]
    if file_extension in pickle_file_extensions:
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        return model
    if file_extension == ".h5":
        tf.compat.v1.disable_eager_execution()
        model = load_model(model_file)
        return model
    if file_extension == ".joblib":
        print("loading joblib model")
        return joblib.load(model_file)

'''
    This function reads the factsheet into a dictionary
'''
def read_factsheet(solution_set_path):
    factsheet_path = os.path.join(solution_set_path, FACTSHEET_NAME)
    if os.path.isfile(factsheet_path):
        with open(factsheet_path,'rb') as f:
            factsheet = json.loads(f.read())
        return factsheet
    else:
        return {}

def read_solution(solution_set_path):
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
            style={"float": "right", "backgroundColor": SECONDARY_COLOR}
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

def load_scenario_description(scenario_path):
    scenario_description = ""
    path = os.path.join(scenario_path, SCENARIO_DESCRIPTION_FILE)
    if os.path.exists(path):
        file = open(path, mode='r')
        scenario_description = file.read()
        file.close()
    return scenario_description
 
def load_scenario_link(scenario_path):
    scenario_link = ""
    path = os.path.join(scenario_path, SCENARIO_LINK_FILE)
    if os.path.exists(path):
        file = open(path, mode='r')
        scenario_link = file.read()
        file.close()
    return scenario_link
    
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
        elif 'pkl' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_pickle(io.BytesIO(decoded))
        df = df[:8]
        
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ]), []
    
    table = html.Div([
        html.H5("Preview of "+filename, className="text-center", style={"color":"DarkBlue"}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'scroll'},
            style_header={"textTransform": "none"},
        ),
        html.Hr(),
    ])
    columns = df.columns.values
    return table, columns

def list_of_metrics(pillar):
    metrics = []
    with open(os.path.join(METRICS_CONFIG_PATH, "config_{}.json".format(pillar))) as file:
        config_file = json.load(file)
        for metric_name in config_file["parameters"]:
            metric_name = metric_name.split("_", 1)[1]
            metrics.append(metric_name.lower())
    return metrics

def create_metric_details_section(metric_id, i, section_n = 1, is_open=False):
    metric_name = metric_id.replace("_", " ")
    return html.Div([

        html.Div([
            html.I(className="fas fa-chevron-down ml-4", id="toggle_{}_details".format(metric_id), style={"float": "right"}),
            html.H4("(X/5)", id="{}_score".format(metric_id), style={"float": "right"}), 
        html.H4("{2}.{0} {1}".format(i+1, metric_name, section_n)),
        ]),
            dbc.Collapse(
            html.Div([NO_DETAILS]),
            id="{}_details".format(metric_id),
            is_open=is_open,          
        ),
        ], id="{}_section".format(metric_id), className="mb-5 mt-5")
        