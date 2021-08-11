import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import json
import os

from app import server
from app import app

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead

from dash.dependencies import Input, Output, State

problem_set_name_input_form = dbc.Form([
    dbc.FormGroup([
        dbc.Label("Name", html_for="problem_set_name"),
        dbc.Input(type="text", id="problem_set_name", placeholder="", debounce=True),
        dbc.FormText(
            "A problem set acts as a container for multiple related models",
            color="secondary",
        ),
        dbc.Button(
            "Submit", id="submit", className="ml-auto", n_clicks=0, style={"float": "right"}
        )
    ])
])

modal = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-plus-circle"),
            id="open", 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Create new problem set"),
                dbc.ModalBody(problem_set_name_input_form),
            ],
            id="modal",
            is_open=False,
        ),
    ]
)

# --- Callbacks --- #
@app.callback(
    Output("problem_set_name", "value"),
    [Input('submit', 'n_clicks')],
    [State('problem_set_name', 'value')])
def create_problem_set_name(n_clicks, name):
    if name:
        app.logger.info("problem_set_name: {}".format(name))
        app.logger.info(problem_set_list_2())
        res = os.mkdir("./problem_sets/{}".format(name))
        return ""
    else:
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
    problem_sets = [(f.name, f.path) for f in os.scandir('./problem_sets') if f.is_dir()]
    problem_set_names = [i[0] for i in problem_sets]
    solution_sets = []
    for name, path in problem_sets:
        solution_set = [f.name for f in os.scandir(path) if f.is_dir()]
        solution_sets.append(solution_set)
    
    final_tree = []
    for i in range(len(problem_set_names)):
        final_tree.append(html.H3(problem_set_names[i]))
        for j in range(len(solution_sets[i])):
            final_tree.append(html.H5("-" + solution_sets[i][j]))
    return final_tree  

def problem_set_list_2():
    problem_sets = [(f.name, f.path) for f in os.scandir('./problem_sets') if f.is_dir()]
    problem_set_names = [i[0] for i in problem_sets]
    solution_sets = []
    for name, path in problem_sets:
        solution_set = [f.name for f in os.scandir(path) if f.is_dir()]
        solution_sets.append(solution_set)
    
    final_tree = []
    for i in range(len(problem_set_names)):
        #final_tree.append(html.H3(problem_set_names[i]))
        for j in range(len(solution_sets[i])):
            final_tree.append({'label': problem_set_names[i] + ' - ' + solution_sets[i][j], 'value': 'path'})
    return final_tree   


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.Div(
                    children=[ 
                        modal,
                        html.H1("Problem Sets", className="text-center"),
                        html.Div(children=problem_set_list()),
                        html.H1(problem_set_list_2())
                    #for problem_set_name, solution_sets in problem_set_dict.items():
                        #html.H5(problem_set_name),
                              
                   ]
                ),
                className="mb-5 mt-5"
            ),
        ])
    ])

])