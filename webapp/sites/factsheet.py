import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import json
import pickle
import os
import io
import base64
from app import app
from config import SCENARIOS_FOLDER_PATH

layout = dbc.Container([
    dbc.Col([], className="mb-4"),
    
    dbc.Col([
        html.Div([
            html.H3("2. Solution"),
        ], 
        className="text-center"
        ),
    ], 
    className="mb-4"
    ),
    
    html.Div(dbc.Button("Download",  id='download_factsheet', color="primary", className="mt-3"), className="text-center"),
    
],
fluid=False
)



if __name__ == '__main__':
    app.run_server(debug=True)

