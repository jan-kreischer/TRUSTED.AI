import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import json
import os
import shutil
from config import SCENARIOS_FOLDER_PATH
from helpers import list_of_scenarios
scenario_dropdown_options = list_of_scenarios()

from app import server
from app import app

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead

from dash.dependencies import Input, Output, State

@app.callback(
    Output("create_scenario_dialog", "is_open"),
    [Input("open_create_scenario_dialog", "n_clicks"), Input("submit_create_scenario_dialog", "n_clicks")],
    [State("create_scenario_dialog", "is_open")],
)
def toggle_create_scenario_dialog(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

create_scenario_dialog = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-plus-circle"),
            id="open_create_scenario_dialog", 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Create a scenario"),
                dbc.ModalBody(
                dbc.Form([
    dbc.FormGroup([
        dbc.Label("Name", html_for="problem_set_name"),
        dbc.Input(type="text", id="problem_set_name", placeholder="", debounce=True),
        dbc.FormText(
            "A scenario acts as a container for multiple different solutions.",
            color="secondary",
        ),
        dbc.Button(
            "Create", id="submit_create_scenario_dialog", className="ml-auto", n_clicks=0, style={"float": "right"}
        )
    ])
])),
            ],
            id="create_scenario_dialog",
            is_open=False,
        ),
    ]
)

@app.callback(
    Output("delete_scenario_dialog", "is_open"),
    [Input("open_delete_scenario_dialog", "n_clicks"), Input("submit_delete_scenario_dialog", "n_clicks")],
    [State("delete_scenario_dialog", "is_open")],
)
def toggle_delete_scenario_dialog(n1, n2, is_open):
    app.logger.info("Open delete")
    if n1 or n2:
        return not is_open
    return is_open

delete_scenario_dialog = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-minus-circle"),
            id="open_delete_scenario_dialog", 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Delete a scenario"),
                dbc.ModalBody(dbc.Form([
    dbc.FormGroup([
        dcc.Dropdown(
                    id='scenario_to_delete',
                    options=scenario_dropdown_options,
                    placeholder='Select Scenario'
        ),
        dbc.Button(
            "Delete", id="submit_delete_scenario_dialog", className="ml-auto", n_clicks=0, style={"float": "right"}
        )
    ])
])),
                
            ],
            id="delete_scenario_dialog",
            is_open=False,
        ),
    ]
)

# --- Callbacks --- #
@app.callback(
    Output("problem_set_name", "value"),
    [Input('submit_create_scenario_dialog', 'n_clicks')],
    [State('problem_set_name', 'value')], prevent_initial_call=True)
def create_scenario(n_clicks, name):
    app.logger.info("Creating scenario {}".format(name))
    if name:
        res = os.mkdir(os.path.join(SCENARIOS_FOLDER_PATH, name))
        return ""
    else:
        return ""
    
# --- Callbacks --- #
@app.callback(
    Output("scenario_to_delete", "value"),
    [Input("submit_delete_scenario_dialog", "n_clicks")],
    [State("scenario_to_delete", 'value')], prevent_initial_call=True)
def delete_scenario(n_clicks, path):
    app.logger.info("Deleting {}".format(path))
    try:
        shutil.rmtree(path, ignore_errors=False)
    except Exception as e:
        print(e)
        raise
    return ""
    

@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("submit", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

def problem_set_list():
    problem_sets = [(f.name, f.path) for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir() and not f.name.startswith('.')]
    problem_set_names = [i[0] for i in problem_sets]
    solution_sets = []
    for name, path in problem_sets:
        solution_set = [f.name for f in os.scandir(path) if f.is_dir()]
        solution_sets.append(solution_set)
    
    final_tree = []
    for i in range(len(problem_set_names)):
        final_tree.append(html.H3(problem_set_names[i], id="scenario_{}".format(problem_set_names[i])))
        for j in range(len(solution_sets[i])):
            final_tree.append(html.H5("-" + solution_sets[i][j]))
    return final_tree  

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.Div(
                    children=[ 
                        delete_scenario_dialog,
                        create_scenario_dialog,
                        html.H1("Scenarios & Solutions", className="text-center"),
                        html.Div(children=problem_set_list()),              
                   ]
                ),
                className="mb-5 mt-5"
            ),
        ])
    ])

])