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
from config import SCENARIOS_FOLDER_PATH
from helpers import create_info_modal

general_inputs = ["model_name", "purpose_description", "domain_description", "training_data_description", "model_information",    "data_normalization", "target_column", "contact_information"]
fairness_inputs = ["protected_feature", "privileged_class_definition"]

@app.callback([Output('create_factsheet_alert', 'children'),
               Output("download_factsheet", "data"),
               Output('model_name', 'value'),
               Output('purpose_description', 'value'),
               Output('domain_description', 'value'),
               Output('training_data_description', 'value'),
               Output('model_information', 'value'), 
               Output('data_normalization', 'value'),
               Output('target_column', 'value'), 
               Output('contact_information', 'value'),
               Output('protected_feature', 'value')],
              [
               Input('download_factsheet_button', 'n_clicks'),
               State('model_name', 'value'),
               State('purpose_description', 'value'),
               State('domain_description', 'value'),
               State('training_data_description', 'value'),
               State('model_information', 'value'), 
               State('data_normalization', 'value'),
               State('target_column_name', 'value'), 
               State('contact_information', 'value'),
               State('protected_feature', 'value')
], prevent_initial_call=True)             
def create_factsheet(
    n_clicks,
    model_name,
    purpose_description,
    domain_description,
    training_data_description,
    model_information,
    data_normalization,
    target_column,
    contact_information,
    protected_feature
):
    print(training_data_description)
    print(n_clicks)
    factsheet = {}
    if n_clicks is not None:
        factsheet["general"] = {}
        for e in general_inputs:
            if eval(e):
                print("{0}, {1}".format(e, eval(e)))
                factsheet["general"][e] = eval(e)
        
        factsheet["fairness"] = {}
        for e in fairness_inputs:
            if eval(e):
                factsheet["fairness"][e] = eval(e)
                
        print(factsheet)
        return html.H3("Created Factsheet", className="text-center", style={"color": "Red"}), dict(content=json.dumps(factsheet), filename="factsheet.json"), "", "", "", "", "", "", "", "", ""
        

for m in general_inputs + fairness_inputs:
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
            html.H1("Factsheet", className="text-center"),
            html.Div([], id="create_factsheet_alert"),
            
            #=== General Information ===
            html.Hr(),
            html.H2("• General Information"),
            #--- Purpose ---
            html.Div([
                create_info_modal("model_name", "Model Name", "Please enter a name for your model.", ""),
                html.H3("Model Name"),
                dcc.Input(id="model_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%'}), 
            ], className="mb-4"),
            
            

            #--- Purpose ---
            html.Div([
                create_info_modal("purpose_description", "Purpose", "Please describe the purpose of your model.", "*e.g Detect multiple objects within an image, with bounding boxes. The model is trained to recognize 80 different classes of objects in the COCO Dataset. The model consists of a deep convolutional net base model for image feature extraction, together with additional convolutional layers specialized for the task of object detection, that was trained on the COCO data set. It is based on SSD MobileNetV1 using the TensorFlow framework.*"),
                html.H3("Purpose"),
                dcc.Textarea(
                    id='purpose_description',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Domain ---
            html.Div([
                create_info_modal("domain_description", "Domain", "Please describe the domain your model is intended to be used in.", "e.g. *The model is designed for the computer vision domain. It can detect 80 different classes of objects like person, bicycle, car, etc. Note: only the \'thing\' category is included. \'Thing\' categories include objects for which individual instances may be easily labeled (person, chair, car).*"),
                html.H3("Domain"),
                dcc.Textarea(
                    id='domain_description',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Training Data ---
            html.Div([
                create_info_modal("training_data_description", "Training Data", "Please describe the training data used to train the model.", "e.g *the model is trained on the COCO dataset. The dataset used in training the model was released in 2015. The number of object categories and the number of instances per category of the MS COCO dataset in comparison with other popular datasets like ImageNet, PASCAL VOC 2012, and SUN is much higher. MS COCO has fewer categories than ImageNet and SUN but has more instances per category which will be useful for learning complex models capable of precise localization. In comparison to PASCAL VOC, MS COCO has both more categories and instances.*"),
                html.H3("Training Data"),
                dcc.Textarea(
                    id='training_data_description',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Training Data ---
            html.Div([
                create_info_modal("model_information", "Model Information", "Please enter the most important information regarding your model", "e.g *The model is based on the SSD MobileNet V1 for TensorFlow. Pre-trained model weights for the model can be found here. SSD stands for Single Shot Detector and a detailed explanation about this architecture can be found here. MobileNet is used as a base network for feature extraction and its architectural details can be found here.*"),
                html.H3("Model Information"),
                dcc.Textarea(
                    id='model_information',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            #--- Data Normalization ---
            html.Div([
                create_info_modal("data_normalization", "Data Normalization", "Please select the normalization technique you used to prepare your data", ""),
                html.H3("Data Normalization"),
                dcc.Dropdown(
                    id='data_normalization',
                    options=[
                        {'label': 'None', 'value': 'none'},
                        {'label': 'Normalization (Min-Max Scaling)', 'value': 'normalization'},
                        {'label': 'Standardization (Z-score Normalization)', 'value': 'standardization'}
                    ],
                    value='none'
            )], className="mb-4 mt-4"),
                        
            #--- Target Column Name ---
            html.Div([
                create_info_modal("target_column", "Target Column Name", "Please enter the name of the target column within your dataset.", ""),
                html.H3("Target Column Name"),
                dcc.Input(id="target_column", type="text", placeholder="", value="", debounce=True, style={'width': '100%'}),
            ], className="mb-4 mt-4"),
                
            #--- Domain ---
            html.Div([
            create_info_modal("contact_information", "Contact Information", "Please enter some contact information in case someone needs help with using your model", ""),
            html.H3("Contact Information"),
            dcc.Textarea(
                id='contact_information',
                value='',
                style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            
            html.Hr(),
            html.H2("• Fairness"),
            html.Div([
                create_info_modal("protected_feature", "Protected Feature", "Please enter the name of the target column within your dataset.", ""),
                html.H3("Protected Feature"),
                dcc.Input(id="protected_feature", type="text", placeholder="", value="", debounce=True, style={'width': '100%'}),
            ], className="mb-4 mt-4"),
            
            html.Div([
                create_info_modal("privileged_class_definition", "Privileged Class Definition", "Please enter the name of the target column within your dataset.", ""),
                html.H3("Privileged Class Definition"),
                dcc.Input(id="privileged_class_definition", type="text", placeholder="e.g lambda x: x >= 25", value="", debounce=True, style={'width': '100%'}),
            ], className="mb-4 mt-4"),
    ], 
    className=""
    ),
    
    html.Div(
        dbc.Button("Download",  id='download_factsheet_button', color="primary", className="mt-3"), className="text-center"
    ),
        
    dcc.Download(id="download_factsheet"),
])
], fluid=False)


if __name__ == '__main__':
    app.run_server(debug=True)

