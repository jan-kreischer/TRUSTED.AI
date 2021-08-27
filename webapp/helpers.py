import os
from config import SCENARIOS_FOLDER_PATH
import glob
import pickle
import pandas as pd
import json
import base64
import io
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from config import TRAINING_DATA_FILE_NAME_REGEX, TEST_DATA_FILE_NAME_REGEX, MODEL_REGEX
from tensorflow.keras.models import load_model
import tensorflow as tf

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

def compute_train_test_split(solution_set_path):
    train_dataset = read_train(solution_set_path)
    test_dataset = read_test(solution_set_path)
    print("TRAIN TEST SPLIT => TRAIN {0}, TEST {1}, SPLIT {2}".format(len(train_dataset), len(test_dataset), len(train_dataset)/len(test_dataset)))
    print("Lenght of test dataset {}".format(len(test_dataset)))
    print("Lenght of training dataset {}".format(len(train_dataset)))
    return len(train_dataset)/len(test_dataset)

def score_train_test_split():
    return 2.5

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