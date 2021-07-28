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
                                                  href="/upload_train_data",
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
                dbc.Col(html.H1("How it works?"), width=12),
                dbc.Col(html.Div([
                    html.Img(src=app.get_asset_url('upload.svg'), height="32px"), 
                    html.Br(),
                    "One of three columns"
                ]), 
                width=3),
                dbc.Col(html.Div("One of three columns"), width=3),
                dbc.Col(html.Div("One of three columns"), width=3),
            ],
            style={
                "background-color": "#EEEEEE",
                "height":  "300px"
            }      
        ),

    
        dbc.Row([
            dbc.Col(html.H1("How it works?"), width=12, className="mb-5 mt-5"),
        ],
        justify="center",
        className="mb-5"),
    
        dbc.Row([
            dbc.Col(html.H1("How it works?"), width=12, className="mb-5 mt-5"),
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


