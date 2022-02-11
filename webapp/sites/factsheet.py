import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import json
import pickle
import os
import io
import base64
from app import app
from config import *
from helpers import create_info_modal

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
def get_factsheet_callbacks(app):
    @app.callback(
           [Output('factsheet_form', 'style'),
            Output('create_factsheet', 'children'),
            Output('factsheet_info', 'children'),
            Output('factsheet_upload', 'style'),
            ],
           Input('create_factsheet', 'n_clicks'))
    def show_factsheet_form(nclicks):
        if nclicks:
            if nclicks % 2 == 1:
                return [{'display': 'block'}, "Upload Factsheet", "Please create a factsheet or upload one using the button",
                        {'display': 'none'} ]
            else:
                return [{'display': 'none'}, "Create Factsheet", "Please upload the factsheet or create a new one using the button",
                        {
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'backgroundColor': '#FFFFFF'
                        }
                        ]
        else:
            return [{'display': 'none'}, "Create Factsheet", "Please upload the factsheet or create a new one using the button",
                    {
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                        'backgroundColor': '#FFFFFF'
                    }
                    ]

    @app.callback([Output('create_factsheet_alert', 'children'),
                   Output("download_factsheet", "data"),
                   Output("created_factsheet", "data"),
                   Output('model_name', 'value'),
                   Output('purpose_description', 'value'),
                   Output('domain_description', 'value'),
                   Output('training_data_description', 'value'),
                   Output('model_information', 'value'),
                   Output('regularization', 'value'),
                   Output('authors', 'value'),
                   Output('contact_information', 'value')
                   ],
                  [
                   Input('download_factsheet_button', 'n_clicks'),
                   State('model_name', 'value'),
                   State('purpose_description', 'value'),
                   State('domain_description', 'value'),
                   State('training_data_description', 'value'),
                   State('model_information', 'value'),
                   State('regularization', 'value'),
                   State('authors', 'value'),
                   State('contact_information', 'value'),
    ], prevent_initial_call=True)
    def create_factsheet(
        n_clicks,
        model_name,
        purpose_description,
        domain_description,
        training_data_description,
        model_information,
        regularization,
        authors,
        contact_information
    ):
        factsheet = {}
        if n_clicks is not None:
            factsheet["general"] = {}
            for e in GENERAL_INPUTS:
                if eval(e):
                    factsheet["general"][e] = eval(e)

            #factsheet["fairness"] = {}
            #for e in FAIRNESS_INPUTS:
            #    if eval(e):
            #        factsheet["fairness"][e] = eval(e)
            #        factsheet["fairness"] = {}

            factsheet["accountability"] = {}
            for e in ACCOUNTABILITY_INPUTS:
                if eval(e):
                    factsheet["accountability"][e] = eval(e)
            return html.H3("Factsheet is created and saved for the analysis", className="text-center", style={"color": "Green"}), dict(content=json.dumps(factsheet, indent=4), filename="factsheet.json"), factsheet, "", "", "", "", "", "", "", ""
        return "", "", "", "", "", "", "", "", "", "", ""

    for m in GENERAL_INPUTS + FAIRNESS_INPUTS + ACCOUNTABILITY_INPUTS_INPUTS:
        @app.callback(
            Output("{}_info_modal".format(m), "is_open"),
            [Input("{}_info_button".format(m), "n_clicks"), Input("{}_close".format(m), "n_clicks")],
            [State("{}_info_modal".format(m), "is_open")], prevent_initial_call=True
        )
        def toggle_input_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open
        
layout = dbc.Container([
    dbc.Col([
        html.Div([
            html.Div([
            #=== General Information ===
            html.H4("Create a Factsheet",  className="text-center"),
            #--- Purpose ---
            html.Div([
                create_info_modal("model_name", "Model Name", "Please enter a name for your model.", ""),
                html.H5("Model Name"),
                dcc.Input(id="model_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%'}), 
            ], className="mb-4"),
            
            #--- Purpose ---
            html.Div([
                create_info_modal("purpose_description", "Purpose", "Please describe the purpose of your model.", "*e.g Detect multiple objects within an image, with bounding boxes. The model is trained to recognize 80 different classes of objects in the COCO Dataset. The model consists of a deep convolutional net base model for image feature extraction, together with additional convolutional layers specialized for the task of object detection, that was trained on the COCO data set. It is based on SSD MobileNetV1 using the TensorFlow framework.*"),
                html.H5("Purpose"),
                dcc.Textarea(
                    id='purpose_description',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Domain ---
            html.Div([
                create_info_modal("domain_description", "Domain", "Please describe the domain your model is intended to be used in.", "e.g. *The model is designed for the computer vision domain. It can detect 80 different classes of objects like person, bicycle, car, etc. Note: only the \'thing\' category is included. \'Thing\' categories include objects for which individual instances may be easily labeled (person, chair, car).*"),
                html.H5("Domain"),
                dcc.Textarea(
                    id='domain_description',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Training Data ---
            html.Div([
                create_info_modal("training_data_description", "Training Data", "Please describe the training data used to train the model.", "e.g *the model is trained on the COCO dataset. The dataset used in training the model was released in 2015. The number of object categories and the number of instances per category of the MS COCO dataset in comparison with other popular datasets like ImageNet, PASCAL VOC 2012, and SUN is much higher. MS COCO has fewer categories than ImageNet and SUN but has more instances per category which will be useful for learning complex models capable of precise localization. In comparison to PASCAL VOC, MS COCO has both more categories and instances.*"),
                html.H5("Training Data"),
                dcc.Textarea(
                    id='training_data_description',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Training Data ---
            html.Div([
                create_info_modal("model_information", "Model Information", "Please enter the most important information regarding your model", "e.g *The model is based on the SSD MobileNet V1 for TensorFlow. Pre-trained model weights for the model can be found here. SSD stands for Single Shot Detector and a detailed explanation about this architecture can be found here. MobileNet is used as a base network for feature extraction and its architectural details can be found here.*"),
                html.H5("Model Information"),
                dcc.Textarea(
                    id='model_information',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Authors ---
            html.Div([
                create_info_modal("authors", "Authors", "Please enter the authors name", ""),
                html.H5("Authors"),
                dcc.Input(id="authors", type="text", placeholder="", value="", debounce=True, style={'width': '100%'})], className="mb-4"),
                
            #--- Contact Information ---
            html.Div([
            create_info_modal("contact_information", "Contact Information", "Please enter some contact information in case someone needs help with using your model", ""),
            html.H5("Contact Information"),
            dcc.Textarea(
                id='contact_information',
                value='',
                style={'width': '100%', 'height': 150},
            )], className="mb-4"),

            html.Div([
            create_info_modal("regularization", "Regularization", "Please select the regularization technique used during training", ""),
            html.H5("Regularization"),
            dcc.Dropdown(
                id='regularization',
                options=[
                    {'label': 'None', 'value': 'none'},
                    {'label': 'Lasso regression (L1)', 'value': 'lasso_regression'},
                    {'label': 'Ridge regression (L2)', 'value': 'ridge_regression'},
                    {'label': 'ElasticNet regression', 'value': 'elasticnet_regression'},
                    {'label': 'Other', 'value': 'Other'}
                ],
                value='none'
            )], className="mb-4 mt-4"),

            html.Div(
                    dbc.Button("Save the Factsheet and Download For Later Use", id='download_factsheet_button', color="primary", className="mt-3"),
                    className="text-center"
            ),
            dcc.Store(id='created_factsheet'),

            dcc.Download(id="download_factsheet"),
            html.Div([], id="create_factsheet_alert"),

            ], style={"border": "1px solid #d8d8d8", "borderRadius": "6px", "backgroundColor": SECONDARY_COLOR}, className="pt-3 pb-3 pl-3 pr-3 mb-4"),
                           
        ],
    className=""
    ),
    

])
],id= "factsheet_form", style= {'display': 'none'}, fluid=False)

if __name__ == '__main__':
    app.run_server(debug=True)

