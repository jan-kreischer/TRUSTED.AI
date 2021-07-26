import dash_html_components as html
import dash_bootstrap_components as dbc

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Quantifying the Trustworthiness Level of Artificial Intelligence", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([dbc.Col(html.H5(children=''), className="mb-4")]),
        dbc.Row([
         dbc.Col(dbc.Card(children=[html.H3(children='Try the Demo',
                                               className="text-center"),
                                       dbc.Button("Demo",
                                                  href="/upload_train_data",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4"),
        dbc.Col(dbc.Card(children=[html.H3(children='Access to the github repo',
                                               className="text-center"),
                                       dbc.Button("Github",
                                                  href="https://github.com/JoelLeupp/Trusted-AI",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4")],
         className="mb-5"),
    ])

])
