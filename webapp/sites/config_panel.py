import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import time
import random
import numpy as np
import pickle
import statistics
import seaborn as sn
import pandas as pd
import json
from math import pi
import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sites.algorithm.helper_functions import get_performance_table, get_final_score, get_case_inputs, trusting_AI_scores
from sites.algorithm.explainability_panel import explainability_panel, exp_input_ids
from sites.algorithm.fairness_panel import fairness_panel ,fair_input_ids
from sites.algorithm.robustness_panel import robustness_panel, rob_input_ids
from sites.algorithm.methodology_panel import methodology_panel, meth_input_ids
import dash_daq as daq

children=[]

# load case inputs
# model, train_data, test_data = get_case_inputs(case)

config_fairness, config_explainability, config_robustness, config_methodology = 0, 0, 0 ,0
for config in ["config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open("sites/algorithm/"+config+".json") as file:
            exec("%s = json.load(file)" % config)

#panels
exp_panel_comp = [html.H3("Explainability Panel", style={'text-align':'center'}),html.Br(),] + explainability_panel
exp_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10,"background-color":"lightyellow"})

fair_panel_comp = [html.H3("Fairness Panel", style={'text-align':'center'}),html.Br(),] + fairness_panel
fair_panel = html.Div(fair_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})

rob_panel_comp = [html.H3("Robustness Panel", style={'text-align':'center'}),html.Br(),] + robustness_panel
rob_panel = html.Div(rob_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})

meth_panel_comp = [html.H3("Methodology Panel", style={'text-align':'center'}),html.Br(),] + methodology_panel
meth_panel = html.Div(meth_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})


children.append(html.Div([html.H3("Configuration",style={'text-align':'center'}),exp_panel,fair_panel,rob_panel,meth_panel,html.Hr()],
                          style={"background-color":"lightyellow"}))

button_div = html.Div([
    html.Hr(),
    html.Div([
                    html.Div(html.Label("Choose Configuration:"), style={'width': '200px', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                    html.Div(dcc.Dropdown(
                                id='config-dropdown',
                                options=list(map(lambda name:{'label': name[:-5], 'value': name} ,os.listdir("configs"))),
                                value='default.json'
                            ), 
                             style={'width': '300px', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                        ]),
    html.Button('apply config', id='apply-config', style={"background-color": "gold",'margin-left': 50}),
    html.Br(),html.Br(),
    html.Button('Save Weights', id='save-weights', style={"background-color": "green",'margin-left': 50}),
    dcc.Store(id='input-config'),
    html.Div(dcc.Input(id="hidden-trigger", value=None, type='text'), style={"display":"none"}),
    # html.Div(id="hidden-trigger-save", style={"display":"none"}),
    html.Br()], style={"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20})
    


children.insert(0,button_div)

panel_div = html.Div(children, id= "panel", style= {'display': 'block'})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#layout = html.Div(children=children, style={"margin-left": "100px","margin-right": "50px"})

#list(filter(lambda ids: ids[:2]=="w_", input_ids))

layout = html.Div([
    panel_div,
    dbc.Modal(
    [   
        dbc.ModalHeader("Save Configuration"),
        dbc.ModalBody([
                       html.Div([
                        html.Div(html.Label("Please enter a name:"), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                        html.Div(dcc.Input(id="config-name", type='text', placeholder="Alias for Configuration", style={"width":"200px"}), 
                                 style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
                        ])
                      ]),
        dbc.ModalFooter(
                    dbc.Button(
                        "Save", id="save-success", className="ml-auto", n_clicks=0, 
                        style={'background-color': 'green','font-weight': 'bold'}
                    )
                ),
    ],
    id="modal-success",
    is_open=False,
    backdrop=True
    ),
    dbc.Modal(
    [
        dbc.ModalHeader("Success"),
        dbc.ModalBody([dbc.Alert(id="alert-success",children ="You successfully saved the configuration", color="success"),
                      ]),
    ],
    id="modal-saved",
    is_open=False,
    backdrop=True
    ),
    daq.BooleanSwitch(id='toggle-hide',
                      on=False,
                      label="show configuration",
                      labelPosition="top",
                      color = "green"
                     
                    ),  
        ], style={"margin-left": "100px","margin-right": "50px"})

input_ids = exp_input_ids + fair_input_ids + rob_input_ids + meth_input_ids

def get_callbacks(app):
    @app.callback([Output('input-config', 'data'), Output("hidden-trigger", "value")], 
              [Input('save-weights', 'n_clicks'),
               Input('apply-config', 'n_clicks')],
                list(map(lambda inp: State(inp, "value"),list(filter(lambda ids: ids[:2]=="w_", input_ids)))))
    def store_input_config(n1, n2, *args):
        
        
        ctx = dash.callback_context
        
        inputs= dict()
        
        for name, val in zip(list(filter(lambda ids: ids[:2]=="w_", input_ids)), args):
            inputs[name[2:]] = float(val)
        
        config_file =  {
            "fairness": {
                "parameters": {
                    "score_Statistical_Parity": {
                        "parameter_XY": {
                            "value": "The value of the parameter_XY",
                            "description": "The description of the paramter and its impact"
                        }
                    }
                },
                "weights": {
                    "Statistical_Parity":  inputs["Statistical_Parity"],
                    "Disparate_Mistreatment":  inputs["Disparate_Mistreatment"],
                    "Class_Imbalance":  inputs["Class_Imbalance"],
                    "Biased_Data":  inputs["Biased_Data"],
                    "Disparate_Treatment":  inputs["Disparate_Treatment"],
                    "Disparate_Impact":  inputs["Disparate_Impact"]
                }
            },
            "explainability": {
                "parameters": {
                    "score_Algorithm_Class": {
                        "clf_type_score": {
                            "value": {
                                "RandomForestClassifier": 3,
                                "KNeighborsClassifier": 3,
                                "SVC": 2,
                                "GaussianProcessClassifier": 3,
                                "DecisionTreeClassifier": 4,
                                "MLPClassifier": 1,
                                "AdaBoostClassifier": 3,
                                "GaussianNB": 3.5,
                                "QuadraticDiscriminantAnalysis": 3,
                                "LogisticRegression": 3,
                                "LinearRegression": 3.5
                            },
                            "description": "Mapping of Learning techniques to the level of explainability based on on literature research and qualitative analysis of each learning technique. For more information see gh-pages/explainability/taxonomy"
                        }
                    },
                    "score_Feature_Relevance": {
                        "scale_factor": {
                            "value": 1.5,
                            "description": "Used for the calculation to detect outliers in a dataset with the help of quartiels and the Interquartile Range (Q3-Q1) for example the lower bound for outliers is then calculated as follows: lw = Q1-scale_factor*IQR"
                        },
                        "distri_threshold": {
                            "value": 0.6,
                            "description": "Used for the calulation of how many features make up the a certain fraction (distri_threshold) of all importance. For example if the distri_threshold is 0.6 and the result would be 10% than this would mean that 10% of the used features concentrate 60% of all feature importance, which would mean that the importance of the features is not well balanced where only a few features are important for the classification and the majority of features has only very little or no impact at all"
                        }
                    }
                },
                "weights": {
                    "Algorithm_Class": inputs["Algorithm_Class"],
                    "Correlated_Features":  inputs["Correlated_Features"],
                    "Model_Size":  inputs["Model_Size"],
                    "Feature_Relevance": inputs["Feature_Relevance"]
                }
            },
            "robustness": {
                "parameters": {
                    "score_Confidence_Score": {
                        "parameter_XY": {
                            "value": "The value of the parameter_XY",
                            "description": "The description of the paramter and its impact"
                        }
                    }
                },
                "weights": {
                    "Confidence_Score":  inputs["Confidence_Score"],
                    "Clique_Method": inputs["Clique_Method"],
                    "Loss_Sensitivity":  inputs["Loss_Sensitivity"],
                    "CLEVER_Score":  inputs["CLEVER_Score"],
                    "Empirical_Robustness_Fast_Gradient_Attack":  inputs["Empirical_Robustness_Fast_Gradient_Attack"],
                    "Empirical_Robustness_Carlini_Wagner_Attack": inputs["Empirical_Robustness_Carlini_Wagner_Attack"],
                    "Empirical_Robustness_Deepfool_Attack":  inputs["Empirical_Robustness_Deepfool_Attack"]
                }
            },
            "methodology": {
                "parameters": {
                    "score_Normalization": {
                        "parameter_XY": {
                            "value": "The value of the parameter_XY",
                            "description": "The description of the paramter and its impact"
                        }
                    }
                },
                "weights": {
                    "Normalization":  inputs["Normalization"],
                    "Treatment_of_Corrupt_Values":  inputs["Treatment_of_Corrupt_Values"],
                    "Train_Test_Split":  inputs["Train_Test_Split"],
                    "Regularization":  inputs["Regularization"],
                    "Treatment_of_Categorical_Features":  inputs["Treatment_of_Categorical_Features"],
                    "Feature_Filtering": inputs["Feature_Filtering"]
                }
            },
            "pillars": {
                "fairness": inputs["fair_pillar"],
                "explainability": inputs["exp_pillar"],
                "robustness": inputs["rob_pillar"],
                "methodology": inputs["meth_pillar"]
            }
        }
        
        return json.dumps(config_file), ctx.triggered[0]['prop_id']
     
    
    @app.callback(
        Output("modal-success", "is_open"),
        [Input("hidden-trigger", "value"),Input("save-success", "n_clicks")],
        State("modal-success", "is_open"))
    def update_output(trig,n, is_open):
        if trig == "save-weights.n_clicks" or n:
            return not is_open
        else:
            return is_open
        
    @app.callback(
        Output("modal-saved", "is_open"),
        [Input("save-success", "n_clicks")],
        [State("modal-saved", "is_open"),
         State("input-config", "data"),
          State("config-name", "value")])
    def save_config(n_clicks, is_open,config, conf_name):
        
        if n_clicks and config is not None:
            config_file = json.loads(config)
            with open('configs/'+ conf_name+'.json', 'w') as outfile:
                json.dump(config_file, outfile, indent=4)
            return not is_open
        else:
            return is_open
        
    # take config and update inputs
    # input_ids = exp_input_ids + fair_input_ids + rob_input_ids + meth_input_ids
    @app.callback(
                list(map(lambda inp: Output(inp, "value"),list(filter(lambda ids: ids[:2]=="w_", input_ids)))), 
                Input('config-dropdown', 'value'))
    def update_config(conf_name):
        
        with open('configs/' + conf_name ,'r') as f:
                config = json.loads(f.read())
                
        output = []
        pillars = ['explainability', 'fairness', 'robustness', 'methodology']
        ids = [exp_input_ids, fair_input_ids, rob_input_ids, meth_input_ids]
        for pillar, pillar_ids in zip(pillars, ids):
            output = output + [config["pillars"][pillar]] + list(map(lambda metric: config[pillar]["weights"][metric[2:]],pillar_ids[1:]))
            
        return output
    