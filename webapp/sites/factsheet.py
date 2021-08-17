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
            html.H1("Factsheet"),
            html.H3("Model Name"),
            dcc.Input(id="model_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%', 'textAlign': 'center'}), 
            
            #--- Purpose ---
            html.H3("Purpose"),
            dcc.Textarea(
                id='purpose_description',
                value='',
                style={'width': '100%', 'height': 150},
            ),
            
            #--- Domain ---
            html.H3("Domain"),
            dcc.Textarea(
                id='domain_description',
                value='',
                style={'width': '100%', 'height': 150},
            ),
            
            #--- Domain ---
            html.H3("Training Data"),
            dcc.Textarea(
                id='training_data_description',
                value='',
                style={'width': '100%', 'height': 150},
            ),
            
            html.H3("Y Column Name"),
            dcc.Input(id="y_column_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%'}),
            
            #--- Domain ---
            html.H3("Contact Information"),
            dcc.Textarea(
                id='contact_information',
                value='',
                style={'width': '100%', 'height': 150},
            ),
        ]
        ),
    ], 
    className=""
    ),
    
    html.Div(dbc.Button("Download",  id='download_factsheet', color="primary", className="mt-3"), className="text-center"),
    
],
fluid=False
)



if __name__ == '__main__':
    app.run_server(debug=True)

