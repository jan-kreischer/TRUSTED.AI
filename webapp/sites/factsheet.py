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


@app.callback(Output('create_factsheet_alert', 'children'),
              [
               Input('create_factsheet_button', 'n_clicks'),
               State('model_name', 'value'),
               State('purpose_description', 'value'),
               State('domain_description', 'value'),
               State('training_data_description', 'value'),
               
], prevent_initial_call=True)             
def create_factsheet(
    n_clicks,
    model_name,
):
    if n_clicks is None:
        return ""
    else:
        if None in (problem_set, model_name, training_data, test_data, model):   
            return html.H5("Please provide all necessary data", style={"color":"Red"},  className="text-center")
        else:
            # Create directory within the problem set to contain the data
            path = problem_set + "/" + model_name
            # Check if directory does not exists yet
            if not os.path.isdir(path):
                os.mkdir(path)
                print("The new directory is created!")
                #return html.H4("Successfully created new directory.", style={"color":"Green"},  className="text-center")
                
                # Upload all the data to the new directory.
                # Saving Training Data
                save_training_data(path, training_data_filename, training_data)
                
                # Saving Test Data
                save_test_data(path, test_data_filename, test_data)
                
                # Saving Factsheet
                save_factsheet(path, "factsheet.json")
                    
                # Saving Model
                save_model(path, model_filename, model)   
            else: 
                return html.H4("Directory already exists", style={"color":"Red"}, className="text-center")
                      
            return dcc.Location(pathname="/analyze", id="someid_doesnt_matter")
            return html.H5("Upload Successful", className="text-center")

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
            
            #--- Training Data ---
            html.H3("Training Data"),
            dcc.Textarea(
                id='training_data_description',
                value='',
                style={'width': '100%', 'height': 150},
            ),
            
            #--- Data Normalization ---
            html.Div([
            html.H3("Data Normalization"),
            dcc.Dropdown(
                id='data_normalization',
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': 'Montreal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                ],
                value='NYC'
            )], className="mb-4 mt-4"),
            
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

