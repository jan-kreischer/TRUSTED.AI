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
exp_panel_comp = [html.H3("Explainability", style={'text-align':'center'})] + explainability_panel
exp_panel = html.Div(exp_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10,"background-color":"lightyellow"})

fair_panel_comp = [html.H3("Fairness", style={'text-align':'center'})] + fairness_panel
fair_panel = html.Div(fair_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})

rob_panel_comp = [html.H3("Robustness", style={'text-align':'center'})] + robustness_panel
rob_panel = html.Div(rob_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})

meth_panel_comp = [html.H3("Methodology", style={'text-align':'center'})] + methodology_panel
meth_panel = html.Div(meth_panel_comp, style={'width': '22%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})


children.append(html.Div([html.H3("Pillars and Metrics\n Weighting",style={'text-align':'center'}),exp_panel,fair_panel,rob_panel,meth_panel],
                          style={"background-color":"lightyellow"}))

button_div = html.Div([
    html.Div(html.Br(), style={'width': '33%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
    html.Div([
    html.Div([html.Br(),
                    html.Div(html.Label("Choose Configuration:"), style={'width': '200px', 'display': 'inline-block',"vertical-align": "top",'margin-left': 50}),
                    html.Div(dcc.Dropdown(
                                id='config-dropdown',
                                options=list(map(lambda name:{'label': name[:-5], 'value': name} ,os.listdir("configs"))),
                                value='default.json'
                            ), 
                             style={'width': '300px', 'display': 'inline-block',"vertical-align": "top",'margin-left': 50}),
                        ]),
    html.Button('apply config', id='apply-config', style={"background-color": "gold",'margin-left': 50}),
    html.Br(),html.Br(),
    html.Button('Save Weights', id='save-weights', style={"background-color": "green",'margin-left': 50}),
    dcc.Store(id='input-config'),
    html.Div(dcc.Input(id="hidden-trigger", value=None, type='text'), style={"display":"none"})],style={'width': '50%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10})
    # html.Div(id="hidden-trigger-save", style={"display":"none"}),
    ], style={"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20," margin-left": "auto", "margin-right": "auto"})
    


children.insert(0, button_div)

panel_div = html.Div(children, id= "panel", style= {'display': 'block'})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#layout = html.Div(children=children, style={"margin-left": "100px","margin-right": "50px"})

#list(filter(lambda ids: ids[:2]=="w_", input_ids))

layout = html.Div([
    #html.Div(
    #[
    #daq.BooleanSwitch(id='toggle-hide',
    #                  on=False,
    #                  label='Show Configuration',
    #                  labelPosition="top",
    #                  color = "green"
    #                 
    #                )],style={"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20}),
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
    ) 
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
            inputs[name] = float(val)
            
        with open('configs/default.json','r') as f:
                config_file = json.loads(f.read())
        
        
        pillars = ['explainability', 'fairness', 'robustness', 'methodology']
        ids = [exp_input_ids, fair_input_ids, rob_input_ids, meth_input_ids]
        for pillar, pillar_ids in zip(pillars, ids):
            #output = output + [config["pillars"][pillar]] + list(map(lambda metric: config[pillar]["weights"][metric[2:]],pillar_ids[1:]))
            config_file["pillars"][pillar] = inputs[pillar_ids[0]]
            for metric in pillar_ids[1:]:
                config_file[pillar]["weights"][metric[2:]] = inputs[metric]
        
        
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
    
    @app.callback(
       Output(component_id='panel', component_property='style'),
       [Input(component_id="show_weighting", component_property='on')])
    def show_hide_element(visibility_state):
        if visibility_state == True:
            return {'display': 'block'}
        if visibility_state == False:
            return {'display': 'none'}