import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Compare", className="text-center"), width=12, className="mb-5 mt-5"),

            dbc.Col([
                dcc.Dropdown(
                    id='demo-dropdown',
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': 'Montreal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    value='Select Problem Set A'
                )], 
                width=6, 
                className="mb-5 mt-5"
            ),
            
            dbc.Col([
                dcc.Dropdown(
                    id='demo-dropdown',
                    options=[
                        {'label': 'New York City', 'value': 'NYC'},
                        {'label': 'Montreal', 'value': 'MTL'},
                        {'label': 'San Francisco', 'value': 'SF'}
                    ],
                    value='Select Problem Set B'
                )], 
                width=6, 
                className="mb-5 mt-5"
            )
        ])
    ])

])
