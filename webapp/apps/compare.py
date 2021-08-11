import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app
import pandas as pd
import os
import json

from .fairness_section import fairness_section
from .explainability_section import explainability_section
from .robustness_section import robustness_section
from .methodology_section import methodology_section

def solution_sets():
    problem_sets = [(f.name, f.path) for f in os.scandir('./problem_sets') if f.is_dir()]
    options = []
    for problem_set_name, problem_set_path in problem_sets:
        solution_sets = [(f.name, f.path) for f in os.scandir(problem_set_path) if f.is_dir()]
        for solution_set_name, solution_set_path in solution_sets:
            options.append({"label": problem_set_name + " > " + solution_set_name, "value": solution_set_path})
    return options

solution_sets = solution_sets()

# === CALLBACKS ===

@app.callback(
    Output('fairness_section', 'children'),
    [Input('solution_set_dropdown_a', 'value')])
def analyze_fairness(solution_set_path):
    if solution_set_path is not None:
        train_data = pd.read_csv("{}/train.csv".format(solution_set_path))
        test_data = pd.read_csv("{}/test.csv".format(solution_set_path))
        
        features = list(train_data.columns)
        y_column_name=""
        factsheet = None
        
        factsheet_path = "{}/factsheet.json".format(solution_set_path)
        # Check if a factsheet.json file already exists in the target directory
        if os.path.isfile(factsheet_path):

            f = open(factsheet_path,)
            factsheet = json.load(f)

            y_column_name = factsheet["y_column_name"]
            app.logger.info(y_column_name)
            protected_column_name = factsheet["protected_column_name"]
            #for i in factsheet:
            #    print(i)
            #app.logger.info(y_column_name)
                
            f.close()
        # Create a factsheet
        else:
            app.logger.info("no factsheet exists yet")
        
        
        solution_set_label_select_options = list(map(lambda x: {"label": x, "value": x}, features))
        solution_set_label_select = html.Div([
            html.H5("Select Label Column"), 
            dcc.Dropdown(
                id='solution_set_label_select',
                options=solution_set_label_select_options,
                value="Credibility"
            ),
        ])
        return [html.H3("Fairness"),solution_set_label_select]
    else:
        return []

    
@app.callback(
    Output(component_id='trust_section', component_property='style'),
    [Input('solution_set_dropdown_a', 'value')])
def show_hide_element(path):
    if path is not None:
        return {'display': 'block'}
    else:
        return {'display': 'none'}
    
@app.callback(
    Output('factsheet_exists', 'children'),
    [Input('solution_set_dropdown_a', 'value')])
def analyze_methdology(solution_set_path):
    if solution_set_path is not None:
        factsheet_path = "{}/factsheet.json".format(solution_set_path)
        if os.path.isfile(factsheet_path):
            return "factsheet provided"
        else:
            return "No factsheet provided"
            
            
@app.callback(
    Output('path_b', 'children'),
    [Input('solution_set_dropdown_b', 'value')])
def update_output_b(value):
    return 'You have selected "{}"'.format(value)

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Compare", className="text-center"), width=12, className="mb-1 mt-1"),

            dbc.Col([
                dcc.Dropdown(
                    id='solution_set_dropdown_a',
                    options=solution_sets,
                    placeholder='Select Model A'
                ),
                html.Div([
                    fairness_section,
                    explainability_section,
                    robustness_section,
                    methodology_section,
                ], id="trust_section", style={"display": "none"})
            ],
                width=6, 
                className="mb-5 mt-1"
            ),
            
            dbc.Col([
                dcc.Dropdown(
                    id='solution_set_dropdown_b',
                    options=solution_sets,
                    placeholder='Select Model B'
                ),
                html.Div(id='path_b')
                ], 
                width=6, 
                className="mb-3 mt-1"
            ),
            

        ])
    ])
])
