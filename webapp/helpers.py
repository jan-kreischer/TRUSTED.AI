import os
import dash_daq as daq
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
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import random
import pickle
import matplotlib.pyplot as plt
from math import pi
import sklearn.metrics as metrics
import collections
from reportlab.pdfgen import canvas
result = collections.namedtuple('result', 'score properties')

def get_performance_metrics(model, test_data, target_column):

    if target_column:
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
    else:
        X_test = test_data.iloc[:,:DEFAULT_TARGET_COLUMN_INDEX]
        y_test = test_data.reset_index(drop=True).iloc[:,DEFAULT_TARGET_COLUMN_INDEX:]
    
    y_true = y_test.values.flatten()
    y_pred = model.predict(X_test).flatten()
    y_pred_proba = model.predict_proba(X_test)
    print("y_true.shape: {}".format(y_true.shape))
    print("y_pred.shape: {}".format(y_pred.shape))
    print("y_pred_proba.shape: {}".format(y_pred_proba.shape))
    #labels = np.unique(np.array([y_pred,y_true]).flatten())

    performance_metrics = pd.DataFrame({
        "accuracy" :  [metrics.accuracy_score(y_true, y_pred)],
        #"global recall" :  [metrics.recall_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted recall" : [metrics.recall_score(y_true, y_pred,average="weighted")],
        #"global precision" : [metrics.precision_score(y_true, y_pred, labels=labels, average="micro")],
        "class weighted precision" : [metrics.precision_score(y_true, y_pred,average="weighted")],
        "global f1 score" :  [metrics.f1_score(y_true, y_pred,average="micro")],
        "class weighted f1 score" :  [metrics.f1_score(y_true, y_pred,average="weighted")],
        #"cross-entropy loss" : [metrics.log_loss(y_true, y_pred_proba)],
        #"ROC AUC" : [metrics.roc_auc_score(y_true, y_pred_proba,average="weighted", multi_class='ovr')]
    }).round(decimals=2)
    return performance_metrics


def get_description(factsheet):
    description = {}
    if "general" in factsheet:
        if "model_name" in factsheet["general"]:
            print("model_name")
            description["Model Name"]= factsheet["general"]["model_name"]
            print(factsheet["general"]["model_name"])
        if "purpose_description" in factsheet["general"]:
            description["Purpose of the Model"] = factsheet["general"]["purpose_description"]
        if "training_data_description" in factsheet["general"]:
            description["Training Data Description"] = factsheet["general"]["training_data_description"]
    description = pd.DataFrame(description, index=[0])
    return description

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

def id_to_name(identifier):
    """This function converts scenario and solution ids into the matching names

    Args:
        n1: number of clicks on the open button.

    Returns:
        Returns false if the dialog was previously open and
        returns true if the dialog was previously closed.

    """
    return identifier.replace("_", " ").title()

def name_to_id(name):
    """This function converts scenario and solution names into a valid ids

    Args:
        n1: number of clicks on the open button.

    Returns:
        Returns false if the dialog was previously open and

    """
    return name.replace(" ", "_").lower()

# === SCENARIOS ===
def get_scenario_ids():
    scenario_ids = [f.name for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir() and not f.name.startswith('.')]
    #sort scenario ids in place
    scenario_ids.sort()
    return scenario_ids

def get_scenario_options():
    scenario_ids = get_scenario_ids()
    options = [{"label": id_to_name(scenario_id), "value": scenario_id} for scenario_id in scenario_ids]
    return options

def get_scenario_path(scenario_id):
    return os.path.join(SCENARIOS_FOLDER_PATH, scenario_id)

def get_solution_ids(scenario_id):
    solution_ids = [(f.name, f.path) for f in os.scandir(get_solution_path(scenario_id, "")) if f.is_dir() and not f.name.startswith('.')]
    solution_ids = sorted(solution_ids, key=lambda x: x[0])
    return solution_ids
    
def get_solution_options():
    scenario_ids = get_scenario_ids()
    options = []
    for scenario_id in scenario_ids:
        scenario_name = id_to_name(scenario_id)
        solutions = get_solution_ids(scenario_id)
        for solution_id, solution_path in solutions:
            solution_name = id_to_name(solution_id)
            options.append({"label": scenario_name + " > " + solution_name, "value": solution_path})
    return options

def get_scenario_solutions_options(scenario_id):
    options = []  
    solutions = get_solution_ids(scenario_id)
    for solution_id, solution_path in solutions:
            solution_name = id_to_name(solution_id)
            options.append({"label": solution_name, "value": solution_path})
    return options

def get_solution_path(scenario_id, solution_id):
    return os.path.join(SCENARIOS_FOLDER_PATH, scenario_id, SOLUTIONS_FOLDER, solution_id)

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

def write_into_factsheet(new_factsheet, solution_set_path):
    factsheet_path = os.path.join(solution_set_path, FACTSHEET_NAME)
    with open(factsheet_path, 'w') as outfile:
        json.dump(new_factsheet, outfile, indent=4)
    return


def save_report_as_pdf(model, test_data, target_column, factsheet):
    l = ["general", "fairness", "methodology"]
    c = canvas.Canvas("report.pdf")
    c.setFont("Times-Roman", 12)
    c.setFillColor('#000080')
    c.drawString(50, 800, "TRUSTED AI")
    c.setStrokeColor('#000080')
    c.setLineWidth(.8)
    c.drawString(280, 750, "REPORT")
    y = 700
    for element in l:
        if factsheet[element] != {}:
            c.setFillColor('#000080')
            y = y - 10
            c.drawString(50, y, id_to_name(element))
            y = y - 15
            c.line(20, y, 580, y)
            y = y - 25
        for k in factsheet[element].keys():
            c.setFillColor('#000000')
            c.drawString(50, y, id_to_name(k)+":")
            c.drawString(200, y, factsheet[element][k])
            y = y - 25

    perf = get_performance_metrics(model, test_data, target_column)
    c.setFillColor('#000080')
    y = y - 10
    c.drawString(50, y, "Performance of the Model")
    y = y - 15
    c.line(20, y, 580, y)
    y = y - 25
    c.setFillColor('#000000')

    for p in perf.columns:
        c.drawString(50, y, id_to_name(p) + ":")
        c.drawString(200, y, perf[p].to_string(index=False))
        y = y - 25
    c.showPage()
    c.save()
    return

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
        for metric_name in config_file["weights"]:
            metrics.append(metric_name.lower())
    return metrics

def create_metric_details_section(metric_id, i, section_n = 1, is_open=False, score="X"):
    metric_name = metric_id.replace("_", " ")
    return html.Div([

        html.Div([
            html.I(className="fas fa-chevron-down ml-4", id="toggle_{}_details".format(metric_id), style={"float": "right"}),
            html.H4("({}/5)".format(score),id="{}_score".format(metric_id), style={"float": "right"}), 
        html.H4("{2}.{0} {1}".format(i+1, metric_name, section_n)),
        ]),
            dbc.Collapse(
            html.Div([NO_DETAILS]),
            id="{}_details".format(metric_id),
            is_open=is_open,          
        ),
        ], id="{}_section".format(metric_id), className="mb-5 mt-5")

def pillar_section(pillar, metrics):
        #configuration_section = []
        #configuration_section.append(html.H3("▶ {} Configuration".format(pillar)))
        #configuration_section.append()
        metric_detail_sections = [html.Div([], id="{}_configuration".format(pillar))]
        for i in range(len(metrics)):
            metric_id = metrics[i].lower()
            metric_detail_sections.append(create_metric_details_section(metric_id, i))

        return html.Div([
                html.Div([
                    dbc.Button(
                        html.I(className="fas fa-chevron-down"),
                        id="toggle_{}_details".format(pillar),
                        className="mb-3",
                        n_clicks=0,
                        style={"float": "right", "backgroundColor": SECONDARY_COLOR}
                    ),
                    daq.BooleanSwitch(id='toggle_{}_mapping'.format(pillar),
                      on=False,
                      label='Show Mappings',
                      labelPosition="top",
                      color = TRUST_COLOR,
                      style={"float": "right"}
                    ),
                    html.H2("• {}".format(pillar.upper()), className="mb-5"),
                    ], id="{}_section_heading".format(pillar.lower())),
                    dbc.Collapse(html.Div(mapping_panel(pillar)[0]),
                        id="{}_mapping".format(pillar),
                        is_open=False,
                        style={"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20, 'display': 'none'}
                    ),
                    html.Br(),
                    html.Div([], id="{}_overview".format(pillar)),
                    html.H3("{0}-Score".format(pillar), className="text-center"),
                    html.Div([], id="{}_star_rating".format(pillar), className="star_rating, text-center"),
                    html.B(["X/5"], id="{}_score".format(pillar), className="text-center", style={"display": "block","font-size":"32px"}),
                    dcc.Graph(id='{}_spider'.format(pillar), style={'display': 'none'}),
                    dcc.Graph(id='{}_bar'.format(pillar), style={'display': 'block'}),    
                    dbc.Collapse(metric_detail_sections,
                        id="{}_details".format(pillar),
                        is_open=False,
                    ),
                    html.Hr(style={"size": "10"}),
                    dbc.Modal(
                    [   
                        dbc.ModalHeader("Save {} Mapping".format(pillar)),
                        dbc.ModalBody([
                                       html.Div([
                                        html.Div(html.Label("Please enter a name:"), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                                        html.Div(dcc.Input(id="mapping-name-{}".format(pillar), type='text', placeholder="Alias for mapping", style={"width":"200px"}), 
                                                 style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                                        ])
                                      ]),
                        dbc.ModalFooter(
                                    dbc.Button(
                                        "Save", id="save-{}-mapping".format(pillar), className="ml-auto", n_clicks=0, 
                                        style={'background-color': 'green','font-weight': 'bold'}
                                    )
                                ),
                    ],
                    id="modal-{}-mapping".format(pillar),
                    is_open=False,
                    backdrop=True
                    ),
                    dbc.Modal(
                    [
                        dbc.ModalHeader("Success"),
                        dbc.ModalBody([dbc.Alert(id="alert-success",children ="You successfully saved the Mapping", color="success"),
                                      ]),
                    ],
                    id="modal-saved-{}".format(pillar),
                    is_open=False,
                    backdrop=True
                    )                   

                ], id="{}_section".format(pillar), style={"display": "None"})
    
def mapping_panel(pillar):
    
    with open('configs/mappings/{}/default.json'.format(pillar), 'r') as f:
                mapping  = json.loads(f.read())
    
    map_panel = []
    input_ids = []
    
    #weight panel
    map_panel.append(html.H4("Mappings",style={'text-align':'center'}))
    # map_panel.append(dcc.Store(id='{}-mapping'.format(pillar)))
    for metric, param in mapping.items():
        map_panel.append(html.H5(metric.replace("_",' '),style={'text-align':'center'}))
        for p, v in param.items():
            input_id = "{}-{}".format(metric,p)
            
            input_ids.append(input_id)
           
            map_panel.append(html.Div(html.Label(v.get("label", p).replace("_",' '), title=v.get("description","")), style={"margin-left":"30%"})),
            if p== "clf_type_score":
                map_panel.append(html.Div(dcc.Textarea(id=input_id, name=pillar,value=str(v.get("value" "")).replace(",",',\n'), style={"width":300, "height":250}), style={"margin-left":"30%"}))
            else:
                map_panel.append(html.Div(dcc.Input(id=input_id, name=pillar,value=str(v.get("value" "")), type='text', style={"width":200}), style={"margin-left":"30%"}))
            map_panel.append(html.Br())
    map_panel.append(html.Hr())      
    map_panel.append(html.Div([html.Label("Load saved mappings",style={"margin-left":10}),
                            dcc.Dropdown(
                                id='mapping-dropdown-{}'.format(pillar),
                                options=list(map(lambda name:{'label': name[:-5], 'value': "configs/mappings/{}/{}".format(pillar,name)} ,os.listdir("configs/mappings/{}".format(pillar)))),
                                value='configs/mappings/{}/default.json'.format(pillar),
                                style={'width': 200},
                                className = pillar
                            )],style={"margin-left":"30%","margin-bottom":15}))
                            
                        
    map_panel.append(html.Div(html.Button('Apply', id='apply-mapping-{}'.format(pillar), style={"background-color": "gold","margin-left":"30%","margin-bottom":15,"width":200})
                     , style={'width': '100%', 'display': 'inline-block'}))
    map_panel.append(html.Div(html.Button('Save', id='save-mapping-{}'.format(pillar), style={"background-color": "green","margin-left":"30%","width":200})
                     , style={'width': '100%', 'display': 'inline-block'}))
    return map_panel , input_ids

def metrics_list(metrics):
    elements = []
    for metric_id in metrics:
        metric_name = id_to_name(metric_id)
        elements.append(html.Li(metric_name, className="text-left"))
    return html.Ul(elements)
        
