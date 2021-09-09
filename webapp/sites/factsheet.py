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

@app.callback([Output('create_factsheet_alert', 'children'),
               Output("download_factsheet", "data"),
               Output('model_name', 'value'),
               Output('purpose_description', 'value'),
               Output('domain_description', 'value'),
               Output('training_data_description', 'value'),
               Output('model_information', 'value'), 
               Output('data_normalization', 'value'),
               Output('regularization', 'value'),
               Output('target_column', 'value'), 
               Output('authors', 'value'), 
               Output('contact_information', 'value'),
               Output('protected_feature', 'value'),
               Output('protected_group', 'value'),
               Output('favorable_outcome', 'value'),
               Output('missing_data', 'value')],
              [
               Input('download_factsheet_button', 'n_clicks'),
               State('model_name', 'value'),
               State('purpose_description', 'value'),
               State('domain_description', 'value'),
               State('training_data_description', 'value'),
               State('model_information', 'value'), 
               State('data_normalization', 'value'),
               State('regularization', 'value'),
               State('target_column', 'value'), 
               State('authors', 'value'),    
               State('contact_information', 'value'),
               State('protected_feature', 'value'),
               State('protected_group', 'value'),
               State('favorable_outcome', 'value'),
               State('missing_data', 'value')
], prevent_initial_call=True)             
def create_factsheet(
    n_clicks,
    model_name,
    purpose_description,
    domain_description,
    training_data_description,
    model_information,
    data_normalization,
    regularization,
    target_column,
    authors,
    contact_information,
    protected_feature,
    protected_group,
    favorable_outcome,
    missing_data
):
    factsheet = {}
    if n_clicks is not None:
        factsheet["general"] = {}
        for e in GENERAL_INPUTS:
            if eval(e):
                factsheet["general"][e] = eval(e)
        
        factsheet["fairness"] = {}
        for e in FAIRNESS_INPUTS:
            if eval(e):
                factsheet["fairness"][e] = eval(e)
                factsheet["fairness"] = {}
                
        factsheet["methodology"] = {}
        for e in METHODOLOGY_INPUTS:
            if eval(e):
                factsheet["methodology"][e] = eval(e)
        return html.H3("Created Factsheet", className="text-center", style={"color": "Red"}), dict(content=json.dumps(factsheet, indent=4), filename="factsheet.json"), "", "", "", "", "", "", "", "", "", "", "", "", "", ""
        

#for m in GENERAL_INPUTS + FAIRNESS_INPUTS + EXPLAINABILITY_INPUTS + ROBUSTNESS_INPUTS + METHODOLOGY_INPUTS:
for m in GENERAL_INPUTS + FAIRNESS_INPUTS:
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
            
            html.Div([
            #=== General Information ===
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
                       
            #--- Target Column Name ---
            html.Div([
                create_info_modal("target_column", "Target Column", "Please enter the name of the target column within your dataset.", ""),
                html.H3("Target Column Name"),
                dcc.Input(id="target_column", type="text", placeholder="", value="", debounce=True, style={'width': '100%'}),
            ], className="mb-4 mt-4"),
            
            #--- Authors ---
            html.Div([
                create_info_modal("authors", "Authors", "Please enter the authors name", ""),
                html.H3("Authors"),
                dcc.Input(id="authors", type="text", placeholder="", value="", debounce=True, style={'width': '100%'})], className="mb-4"),
                
            #--- Contact Information ---
            html.Div([
            create_info_modal("contact_information", "Contact Information", "Please enter some contact information in case someone needs help with using your model", ""),
            html.H3("Contact Information"),
            dcc.Textarea(
                id='contact_information',
                value='',
                style={'width': '100%', 'height': 150},
            )], className="mb-4"),
            ], style={"border": "1px solid #d8d8d8", "borderRadius": "6px", "backgroundColor": SECONDARY_COLOR}, className="pt-3 pb-3 pl-3 pr-3 mb-4"),
                
            html.Div([
            html.H2("• Fairness"),
            html.Div([
                create_info_modal("protected_feature", "Protected Feature", "Please enter the name of the target column within your dataset.", ""),
                html.H3("Protected Feature"),
                dcc.Input(id="protected_feature", type="text", placeholder="", value="", debounce=True, style={'width': '100%'}),
            ], ),
            
            html.Div([
                create_info_modal("protected_group", "Protected Group Definition", "Please enter the name of the target column within your dataset.", ""),
                html.H3("Protected Group Definition"),
                dcc.Input(id="protected_group", type="text", placeholder=PROTECTED_GROUP_DEFINITION_EXAMPLE, value="", debounce=True, style={'width': '100%'}),
            ], className="mb-4 mt-4"),
                
             html.Div([
                create_info_modal("favorable_outcome", "Favorable Outcome", "Please enter a lambda expression defining values of the target column which are seen as favorable.", ""),
                html.H3("Favorable Outcome"),
                dcc.Input(id="favorable_outcome", type="text", placeholder=FAVORABLE_OUTCOME_DEFINITION_EXAMPLE, value="", debounce=True, style={'width': '100%'}),
            ], className="mb-4 mt-4")
            ], style={"border": "1px solid #d8d8d8", "borderRadius": "6px", "backgroundColor": SECONDARY_COLOR}, className="pt-3 pb-3 pl-3 pr-3 mb-4 mt-4"),
                        
            #html.Div([
            #    html.H2("• Explainability")
            #], style={"border": "1px solid #d8d8d8", "borderRadius": "6px", "backgroundColor": SECONDARY_COLOR}, className="pt-3 pb-3 pl-3 pr-3 mb-4"),
            
            #html.Div([
            #
            #    html.H2("• Robustness")
            #], style={"border": "1px solid #d8d8d8", "borderRadius": "6px", "backgroundColor": SECONDARY_COLOR}, className="pt-3 pb-3 pl-3 pr-3 mb-4"),
            
            html.Div([
                html.H2("• Methodology"), 
                #--- Normalization ---
                html.Div([
                    create_info_modal("data_normalization", "Data Normalization", "Please select the normalization technique you used to prepare your data", ""),
                    html.H3("Data Normalization"),
                    dcc.Dropdown(
                        id='data_normalization',
                        options=[
                            {'label': 'None', 'value': 'none'},
                            {'label': 'Normalization (Min-Max Scaling)', 'value': 'normalization'},
                            {'label': 'Standardization (Z-score Normalization)', 'value': 'standardization'},
                            {'label': 'Other', 'value': 'Other'}
                        ],
                        value='none'
                )], className="mb-4 mt-4"),
                
                html.Div([
                    create_info_modal("regression", "Regression", "Please select the regression technique used during training", ""),
                    html.H3("Regularization"),
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
                
                html.Div([
                    create_info_modal("missing_data", "Missing Data", "What technique did you use in order to deal with missing data.", ""),
                    html.H3("Missing Data"),
                dcc.Textarea(
                    id='missing_data',
                    value='',
                    style={'width': '100%', 'height': 150},
            )], className="mb-4 mt-4"),
            ], style={"border": "1px solid #d8d8d8", "borderRadius": "6px", "backgroundColor": SECONDARY_COLOR}, className="pt-3 pb-3 pl-3 pr-3 mb-4"),         
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

