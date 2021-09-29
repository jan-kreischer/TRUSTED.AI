import dash_html_components as html
import dash_bootstrap_components as dbc
from app import server
from app import app
from config import *
from helpers import *

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Quantifying the Trustworthiness Level of Artificial Intelligence", className="text-center"), className="mb-4 mt-4")
        ]),
        dbc.Row([
         dbc.Col(
             dbc.Card(
                 children=[html.H3(children='Try the Demo', className="text-center"),
                                       dbc.Button("Demo",
                                                  href="/upload",
                                                  className="mt-3",
                                                  style={"background-color": TRUST_COLOR, "color": "#FFFFFF", "opacity": "0.8"}
                                           ),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4")
        ],
        justify="center",
        className="mb-5"),
        dbc.Row(
            [   
                dbc.Col(html.H2("How it works?"), width=12, className="text-center mt-4 mb-4"),
                dbc.Col([html.I(className="fas fa-cloud-upload-alt fa-3x", style={"width": "64px"}), html.H3("1. Upload Data", className="mt-3"), html.H5("You are building machine learning models for your organisation \n and you are unsure how trustworthy they are? Use our platform to get answers to your questions. You upload your model, training data and factsheet to our platform.", style={"textTransform": "none"})], width={"size": 2, "order": 1, "offset": 3}, className="text-center"),
                dbc.Col([html.I(className="fas fa-search fa-3x", style={"width": "64px"}), html.H3("2. Analyze", className="mt-3"), html.H5("We analyze your model regarding fairness, explainablity, robustness and methodology using industry standard metrics. This will tell you how well your model performs.", style={"textTransform": "none"})], width={"size": 2, "order": 2, "offset": 0}, className="text-center"),
                dbc.Col([html.I(className="fas fa-level-up-alt fa-3x", style={"width": "64px"}), html.H3("3. Improve", className="mt-3"), html.H5("Your model will be checked in many different dimension. This gives you valuable performance insights. Use this assessment in order to further enhance your model", style={"textTransform": "none"})], width={"size": 2, "order": 3, "offset": 0}, className="text-center"),
            ],
            className="mt-4 mb-4 pt-4 pb-4",
            style={
                "background-color": SECONDARY_COLOR
            }      
        ),

    
        dbc.Row([
            dbc.Col(html.H2("What metrics do we apply?"), width=12, className="text-center"),
            
            dbc.Col(html.Div([html.H3("Fairness"), html.Div("Impartial and just decisions without discrimination of protected groups.", ), metrics_list(FAIRNESS_METRICS)]), width=2, className="text-center"),
            dbc.Col(html.Div([html.H3("Explainability"), html.Div("Provide clarification for the cause of the decision"), metrics_list(EXPLAINABILITY_METRICS)]), width=2, className="text-center"),
            dbc.Col(html.Div([html.H3("Robustness"), html.Div("Resilience against adversarial inputs"), metrics_list(ROBUSTNESS_METRICS)]), width=2, className="text-center"),
            dbc.Col(html.Div([html.H3("Methodology"), html.Div("Quality analysis of the model lifecycle"), metrics_list(METHODOLOGY_METRICS)]), width=2, className="text-center"),
        ],
        justify="center",
        className="mt-4 mb-4 pt-4 pb-4"),
    
        dbc.Row([
        ],
        style={
                "background-color": TRUST_COLOR,
                "height":  "300px",
                "opacity": "0.75",
                "margin-bottom": "0px"
        },   
        justify="center",
        className="mb-5 mt-5"),
    ],
    fluid=True
)


