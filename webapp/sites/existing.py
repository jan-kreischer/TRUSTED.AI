import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import json
import os
import shutil
from config import *
from helpers import *
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
        dbc.FormText(
            "A scenario acts as a container for multiple different solutions.",
            color="secondary",
        ),
        dbc.Label("Name", html_for="scenario_name"),
        dbc.Input(type="text", id="scenario_name", placeholder="", debounce=True),
        dbc.Label("Description", html_for="scenario_name"),
        dcc.Textarea(
            id='scenario_description',
            value='',
            style={'width': '100%', 'height': 100},
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
    [Output("scenario_name", "value"),Output("scenario_description", "value")],
    [Input('submit_create_scenario_dialog', 'n_clicks')],
    [State('scenario_name', 'value'), State('scenario_description', 'value')], prevent_initial_call=True)
def create_scenario(n_clicks, scenario_name, scenario_description):
    if scenario_name:
        res = os.mkdir(os.path.join(SCENARIOS_FOLDER_PATH, scenario_name))
        f = open(os.path.join(SCENARIOS_FOLDER_PATH, scenario_name, SCENARIO_DESCRIPTION_FILE), "w")
        print(scenario_description)
        f.write(scenario_description)
        f.close()
    return "", ""

    
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

def scenario_list():
    scenarios = [(f.name, f.path) for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir() and not f.name.startswith('.')]
    scenario_names = [i[0] for i in scenarios]
    scenario_paths = [i[1] for i in scenarios]
    solution_sets = []
    for name, path in scenarios:
        solution_set = [f.name for f in os.scandir(path) if f.is_dir()]
        solution_sets.append(solution_set)
    
    final_tree = []
    for i in range(len(scenario_names)):
        final_tree.append(html.H3(scenario_names[i], id="scenario_{}".format(scenario_names[i])))
        final_tree.append(html.Div(load_scenario_description(scenario_paths[i]), className="mt-2 mb-4", style={"font-style": "italic"}))
        for j in range(len(solution_sets[i])):
            final_tree.append(html.H5("-" + solution_sets[i][j]))
        final_tree.append(html.Hr())
    return final_tree  

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.Div(
                    children=[ 
                        delete_scenario_dialog,
                        create_scenario_dialog,
                        html.H1("Scenarios", className="text-center"),
                        html.Div(children=scenario_list()),              
                   ]
                ),
                className="mb-5 mt-5"
            ),
        ])
    ])

])