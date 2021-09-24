import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
import os
import pandas as pd
import io
import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_table
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go

from config import *

# must add this line in order for the app to be deployed successfully on Heroku
from app import server
from app import app
# import all pages in the app
from sites import homepage, upload, analyze, compare, scenarios, factsheet #, visualisation 
from sites.config_panel import input_ids 

navbar = dbc.Navbar(
    dbc.Container([
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=app.get_asset_url('logo.svg'), height="32px")),
                        dbc.Col(dbc.NavbarBrand("TRUSTED.AI", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [                
                        dbc.NavItem(dbc.NavLink("Scenarios", href="/scenarios")),
                        dbc.NavItem(dbc.NavLink("Factsheet", href="/factsheet")),
                        dbc.NavItem(dbc.NavLink("Upload", href="/upload")),
                        dbc.NavItem(dbc.NavLink("Analyze", href="/analyze")),
                        dbc.NavItem(dbc.NavLink("Compare", href="/compare"))
                    ], className="ml-auto", navbar=True
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        fluid=False
    ),
    color=PRIMARY_COLOR,
    dark=True,
    className="mb-4",
)

@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page_content'),
    dcc.Store(id='uploaded_solution_set_path', storage_type='session')
])
    
@app.callback(Output('page_content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return homepage.layout
    if pathname == '/upload':
        return upload.layout
    if pathname == '/analyze':
        return analyze.layout
    elif pathname == '/compare':
        return compare.layout
    elif pathname == '/scenarios':
        return scenarios.layout
    elif pathname == '/factsheet':
        return factsheet.layout
    # elif pathname == '/visualisation':
    #     return visualisation.layout
    else:
        return homepage.layout
    
#visualisation.get_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)