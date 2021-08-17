import dash_html_components as html
import dash_bootstrap_components as dbc
from app import server
from app import app

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
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4")
        ],
        justify="center",
        className="mb-5"),
        dbc.Row(
            [   
                dbc.Col(html.H2("How it works?"), width=12, className="text-center"),
                dbc.Col([html.I(className="fas fa-cloud-upload-alt", style={"width": "64px"}), html.H3("Upload Data"), html.H5("You are building machine learning models for your organisation \n and you are unsure how trustworthy they are? Use our platform to get answers to your questions. You upload your model, training data and factsheet to our platform.")], width=4, className="text-center"),
                dbc.Col([html.I(className="fas fa-search", style={"width": "64px"}), html.H3("Analyze"), html.H5("We analyze your model regarding fairness, explainablity, robustness and methodology using industry standard metrics. This will tell you how well your model performs.")], width=4, className="text-center"),
                dbc.Col([html.I(className="fas fa-level-up-alt", style={"width": "64px"}), html.H3("Improve"), html.H5("Your model will be checked in many different dimension. This gives you valuable performance insights. Use this assessment in order to further enhance your model")], width=4, className="text-center"),
            ],
            className="mt-2 mb-2",
            style={
                "background-color": "#EEEEEE",
                "height":  "300px"
            }      
        ),

    
        dbc.Row([
            dbc.Col(html.H2("What metrics do we apply?"), width=12, className="text-center"),
            
            dbc.Col(html.Div(html.H3("Fairness")), width=3, className="text-center"),
            dbc.Col(html.Div(html.H3("Explainability")), width=3, className="text-center"),
            dbc.Col(html.Div(html.H3("Robustness")), width=3, className="text-center"),
            dbc.Col(html.Div(html.H3("Methodology")), width=3, className="text-center"),
        ],
        justify="center",
        className="mb-5"),
    
        dbc.Row([
        ],
        style={
                "background-color": "#1a1a1a",
                "height":  "300px"
        },   
        justify="center",
        className="mb-5 mt-5"),
    
    #1a1a1a
    
    ],
    fluid=True
)


