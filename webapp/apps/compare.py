import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from app import app
import os

def solution_sets():
    problem_sets = [(f.name, f.path) for f in os.scandir('./problem_sets') if f.is_dir()]
    options = []
    for problem_set_name, problem_set_path in problem_sets:
        solution_sets = [(f.name, f.path) for f in os.scandir(problem_set_path) if f.is_dir()]
        for solution_set_name, solution_set_path in solution_sets:
            options.append({"label": problem_set_name + " > " + solution_set_name, "value": problem_set_path + "/" + solution_set_path})
    return options

solution_sets = solution_sets()

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Compare", className="text-center"), width=12, className="mb-1 mt-1"),

            dbc.Col([
                dcc.Dropdown(
                    id='demo-dropdown',
                    options=solution_sets
                )], 
                width=6, 
                className="mb-1 mt-1"
            ),
            
            dbc.Col([
                dcc.Dropdown(
                    id='demo-dropdown',
                    options=solution_sets,
                )], 
                width=6, 
                className="mb-1 mt-1"
            )
        ])
    ])

])
