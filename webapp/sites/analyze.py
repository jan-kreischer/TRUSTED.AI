import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
from app import app
import pandas as pd
import os
import json
import glob

from pillars.fairness.class_balance import compute_class_balance

def solution_sets():
    problem_sets = [(f.name, f.path) for f in os.scandir('./problem_sets') if f.is_dir()]
    options = []
    for problem_set_name, problem_set_path in problem_sets:
        solution_sets = [(f.name, f.path) for f in os.scandir(problem_set_path) if f.is_dir()]
        for solution_set_name, solution_set_path in solution_sets:
            options.append({"label": problem_set_name + " > " + solution_set_name, "value": solution_set_path})
    return options

solution_sets = solution_sets()

FAIRNESS_HIGHLIGHT_COLOR = "yellow"
EXPLAINABLITY_HIGHLIGHT_COLOR = "cornflowerblue"
ROBUSTNESS_HIGHLIGHT_COLOR = "lightgrey"
METHODOLOGY_HIGHLIGHT_COLOR = "lightseagreen"

# === TRUST ===
@app.callback(
    [Output("trust_overview", 'children'),
    Output("trust_details", 'children')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def analyze_trust(solution_set_path):
    if solution_set_path is not None:
        df = pd.DataFrame(dict(
            r=[1, 5, 2, 2],
            theta=['processing cost','mechanical properties','chemical stability',
           'thermal stability']))
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        trust_overview = dcc.Graph(figure=fig)
        return [trust_overview, html.H3("8/10")], [html.H3("8/10")]
    else:
        return [], []


# === FAIRNESS ===
@app.callback(
    [Output("fairness_overview", 'children'),
    Output("fairness_details", 'children')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def analyze_fairness(solution_set_path):
    if solution_set_path is not None:
        train_data = pd.read_csv("{}/train.csv".format(solution_set_path))
        test_data = pd.read_csv("{}/test.csv".format(solution_set_path))

        features = list(train_data.columns)
        y_column_name=""
        factsheet = None

        factsheet_path = "{}/factsheet.json".format(solution_set_path)
        # Check if a factsheet.json file already exists in the target directory
        if os.path.isfile(factsheet_path):

            f = open(factsheet_path,)
            factsheet = json.load(f)

            y_column_name = factsheet["y_column_name"]
            
            app.logger.info(y_column_name)
            protected_column_name = factsheet["protected_column_name"]

            f.close()
        # Create a factsheet
        else:
            app.logger.info("no factsheet exists yet")


        solution_set_label_select_options = list(map(lambda x: {"label": x, "value": x}, features))
        solution_set_label_select = html.Div([
            html.H5("Select Label Column"), 
            dcc.Dropdown(
                id="solution_set_label_select",
                options=solution_set_label_select_options,
                value=y_column_name
            ),
        ])

        fig = px.histogram(train_data, x="Creditability")
        class_balance_graph = dcc.Graph(figure=fig)

        df = pd.DataFrame(dict(
            r=[1, 5, 2, 2, 3],
            theta=['processing cost','mechanical properties','chemical stability',
           'thermal stability', 'device integration']))
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        fairness_overview = dcc.Graph(figure=fig)
        fairness_metrics_class_balance = html.Div("Class Balance", id="fairness_metrics_class_balance")
        return [fairness_overview], [html.H4("Fairness Metrics"), solution_set_label_select, fairness_metrics_class_balance]
    else:
        return [html.H1("Nothing")], [html.H1("Nothing")]

def update_factsheet(factsheet_path, key, value):
    print("update factsheet {0} with {1}  {2}".format(factsheet_path, key, value))
    jsonFile = open(factsheet_path, "r") # Open the JSON file for reading
    data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    ## Working with buffered content
    data[key] = value
    
    print(data)
    ## Save our changes to JSON file
    jsonFile = open(factsheet_path, "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()
     
'''
The following function updates
'''
@app.callback(
    Output("fairness_metrics_class_balance", 'children'),
    [Input("solution_set_label_select", 'value'), 
    State('training_data', 'data'),
    State("solution_set_dropdown", 'value')])
def fairness_metrics_class_balance(label, jsonified_training_data, solution_set_path):
    app.logger.info("Why triggerst du nicht?")
    training_data = pd.read_csv("{}/train.csv".format(solution_set_path))
    graph = dcc.Graph(figure=px.histogram(training_data, x=label, opacity=0.5, title="Label vs Label Occurence", color_discrete_sequence=['#00FF00']))
    app.logger.info("Was mach ich hier eigentlich?")
    #compute_class_balance("hi")
    update_factsheet("{}/factsheet.json".format(solution_set_path), "y_column_name", label)
    return [html.H3("1.1 Class Balance"), graph]
    
    #print("label {}".format(label))
    #print("JSONIFIED TRAINING DATA {}".format(jsonified_training_data))
    #training_data = pd.read_json(jsonified_training_data, orient='split')
    #print("Training data")
    #print(dff.head(5))
    #figure = create_figure(dff)
    #fig = px.histogram(train_data, x=label)
    #graph = dcc.Graph(figure=px.histogram(training_data, x=label))
    #return [html.H3("Selected {} as label column. Computing class balance now.".format(label)), graph]

@app.callback(
    [Output('training_data', 'data'),
     Output('test_data', 'data')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def load_data(solution_set_path):
     # some expensive clean data step
     #cleaned_df = your_expensive_clean_or_compute_step(value)

     # more generally, this line would be
     # json.dumps(cleaned_df)
    app.logger.info("loaded data")
    if solution_set_path is not None:
        training_data = pd.read_csv("{}/train.csv".format(solution_set_path))
        print("LENGTH OF TRAIN DATA {}".format(len(training_data)))
        print(training_data.head(5))

        test_data = pd.read_csv("{}/test.csv".format(solution_set_path))
        print("LENGTH OF TEST DATA {}".format(len(test_data)))
        print(test_data.head(5))

        app.logger.info("SAVED DATA")
        return training_data.to_json(date_format='iso', orient='split'), test_data.to_json(date_format='iso', orient='split'), 
        #return json.dumps(training_data), json.dumps(test_data)
    else:
        return None, None
    
#@app.callback(Output('graph', 'figure'), Input('intermediate-value', 'data'))
#def update_graph(jsonified_cleaned_data):

    # more generally, this line would be
    # json.loads(jsonified_cleaned_data)
#    dff = pd.read_json(jsonified_cleaned_data, orient='split')

#    figure = create_figure(dff)
#    return figure

#@app.callback(Output('table', 'children'), Input('intermediate-value', 'data'))
#def update_table(jsonified_cleaned_data):
#    dff = pd.read_json(jsonified_cleaned_data, orient='split')
#    table = create_table(dff)
#    return table
    
# === EXPLAINABILITY ===
@app.callback(
    [Output("explainability_overview", 'children'),
    Output("explainability_details", 'children')],
    [Input("solution_set_dropdown", 'value')], prevent_initial_call=True)
def analyze_explainability(solution_set_path):
    if solution_set_path is not None:

        df = pd.DataFrame(dict(
            r=[1, 5, 2, 2, 3],
            theta=['processing cost','mechanical properties','chemical stability',
           'thermal stability', 'device integration']))
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        explainability_overview = dcc.Graph(figure=fig)
        return [explainability_overview], []
    else:
        return [], []
        

# === ROBUSTNESS ===
@app.callback(
    [Output("robustness_overview", 'children'),
    Output("robustness_details", 'children')],
    [Input("solution_set_dropdown", 'value')], prevent_initial_call=True)
def analyze_robustness(solution_set_path):
    if solution_set_path is not None:
        df = pd.DataFrame(dict(
            r=[1, 5, 2, 2, 3],
            theta=['processing cost','mechanical properties','chemical stability',
           'thermal stability', 'device integration']))
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        explainability_overview = dcc.Graph(figure=fig)
        return [explainability_overview], [html.Div("Robustness Metrics")]
    else:
        return [], []
 
# === METHODOLOGY ===
'''
for c in columns:
    print("Methodology {}".format(c))
    @app.callback(
        [Output("methodology_overview_{}".format(c), 'children'),
        Output("methodology_details_{}".format(c), 'children')],
        [Input("solution_set_dropdown".format(c), 'value')], prevent_initial_call=True)
    def analyze_methodology(solution_set_path):
        if solution_set_path is not None:
            df = pd.DataFrame(dict(
                r=[1, 5, 2, 2, 3],
                theta=['processing cost','mechanical properties','chemical stability',
               'thermal stability', 'device integration']))
            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
            fig.update_traces(fill='toself')
            explainability_overview = dcc.Graph(figure=fig)
            return [explainability_overview], [html.Div("Methodology Metrics {}".format(c), style={"display": "None"})]
        else:
            return [], []
'''


@app.callback(
    [Output("methodology_overview", 'children'),
    Output("methodology_details", 'children')],
    [Input("solution_set_dropdown", 'value')], prevent_initial_call=True)
def analyze_methodology(solution_set_path):
    if solution_set_path is not None:
        df = pd.DataFrame(dict(
            r=[1, 5, 2, 2, 3],
            theta=['processing cost','mechanical properties','chemical stability',
           'thermal stability', 'device integration']))
        fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        fig.update_traces(fill='toself')
        explainability_overview = dcc.Graph(figure=fig)
        return [explainability_overview], [html.Div("Methodology Metrics a", style={"display": "None"})]
    else:
        return [], []
    
@app.callback(
    Output(component_id="trust_section", component_property='style'),
    Output(component_id="fairness_section", component_property='style'),
    Output(component_id="explainablity_section", component_property='style'),
    Output(component_id="robustness_section", component_property='style'),
    Output(component_id="methodology_section", component_property='style'),
    [Input("solution_set_dropdown", 'value')], prevent_initial_call=True)
def toggle_pillar_section_visibility(path):
    app.logger.info(path)
    app.logger.info("called show hide element")
    if path is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
   
#@app.callback(
#   Output(component_id='element-to-hide', component_property='style'),
#   [Input(component_id='dropdown-to-show_or_hide-element', component_property='value')])
#def show_hide_element(visibility_state):
#    if visibility_state == 'on':
#        return {'display': 'block'}
#    if visibility_state == 'off':
#        return {'display': 'none'}

@app.callback(
    [Output('alert_section', 'children'),
    Output('analysis_section', 'style')],
    [Input('solution_set_dropdown', 'value')])
def analyze_methdology(solution_set_path):
    alerts = []
    style={'display': 'block'}
    if solution_set_path is not None:
        factsheet_path = "{}/factsheet.*".format(solution_set_path)
        test_data_path = "{}/test.*".format(solution_set_path)
        training_data_path = "{}/train.*".format(solution_set_path)
        model_path = "{}/model.*".format(solution_set_path)
        
        if not glob.glob(factsheet_path):
            alerts.append(html.H5("No factsheet provided", className="text-center", style={"color":"Red"}))
        if not glob.glob(training_data_path):
            alerts.append(html.H5("No training data provided", className="text-center", style={"color":"Red"}))
        if not glob.glob(test_data_path):
            alerts.append(html.H5("No test data provided", className="text-center", style={"color":"Red"}))
        if not glob.glob(model_path):
            alerts.append(html.H5("No model provided", className="text-center", style={"color":"Red"}))
        if alerts:
            style={'display': 'none'}
            alerts.append(html.H5("Please provide a complete dataset", className="text-center", style={"color":"Red"}))
    return alerts, style

# === SECTIONS ===
def trust_section(c):
    return html.Div([ 
        html.H2("Trustworthiness"),
        html.Div([], id="trust_overview"),
        html.Div([], id="trust_details"),
        html.Hr()
    ], id="trust_section", style={"display": "None"})

def fairness_section(c):
    return html.Div([
        dbc.Button(
            html.I(className="fas fa-chevron-down"),
            id="toggle_fairness_details",
            className="mb-3",
            n_clicks=0,
            style={"float": "right"}
        ),
        html.H3("1. Fairness"),
        html.Div([], id="fairness_overview"),
        dbc.Collapse([], id="fairness_details", is_open=False),
        html.Hr(),
        #html.Div(id="class_balance"),
    ], id="fairness_section", style={"display": "None"})

def explainability_section(c):
    return html.Div([
        dbc.Button(
            html.I(className="fas fa-chevron-down"),
            id="toggle_explainability_details",
            className="mb-3",
            n_clicks=0,
            style={"float": "right"}
        ),
        html.H3("2. Explainablity"),
        html.Div([
            html.H6("Explainablity Overview")],
            id="explainability_overview"
        ),
        dbc.Collapse(
            html.P("Explainability Details"),
            id="explainability_details",
            is_open=False,
        ),
        html.Hr(),
    ], id="explainablity_section", style={"display": "None"})


def robustness_section(c):
    return html.Div([
        dbc.Button(
            html.I(className="fas fa-chevron-down"),
            id="toggle_robustness_details",
            className="mb-3",
            n_clicks=0,
            style={"float": "right"}
        ),
        html.H3("3. Robustness"),
        html.Div([], id="robustness_overview"),
        dbc.Collapse(
            html.P("Robustness Details"),
            id="robustness_details",
            is_open=False,
        ),
        html.Hr(),
    ], id="robustness_section", style={"display": "None"})


def methodology_section(c):
    return html.Div([
        dbc.Button(
            html.I(className="fas fa-chevron-down"),
            id="toggle_methodology_details",
            className="mb-3",
            n_clicks=0,
            style={"float": "right"}
        ),
        html.H3("4. Methodology"),
        
        html.Div([], id="methodology_overview"),
        dbc.Collapse(
            html.P("Methodology Details"),
            id="methodology_details",
            is_open=False,
            style={"display": "None"}
        ),
        html.Hr(),
    ], id="methodology_section", style={"display": "None"})


def alert_section(c):
    return html.Div([
    ], id="alert_section")


SECTIONS = ['trust', 'fairness', 'explainablity', 'robustness', 'methodology']

for s in SECTIONS:
    @app.callback(
        [Output("{0}_details".format(s), "is_open"),
        Output("{0}_details".format(s), "style")],
        [Input("toggle_{0}_details".format(s), "n_clicks")],
        [State("{0}_details".format(s), "is_open")],
        prevent_initial_call=True
    )
    def toggle_detail_section(n, is_open):
        app.logger.info("toggle {0} detail section".format(s))
        if is_open:
            return (not is_open, {'display': 'None'})
        else:
            return (not is_open, {'display': 'Block'})

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Analyze", className="text-center"), width=12, className="mb-2 mt-1"),
            
            dbc.Col(dcc.Dropdown(
                    id='solution_set_dropdown',
                    options=solution_sets,
                    placeholder='Select Model A'
                ), width=12, style={"marginLeft": "0 px", "marginRight": "0 px"}, className="mb-1 mt-1"
            ),
                     
            dbc.Col([
                alert_section('a'),
                html.Div([
                    trust_section('a'),
                    fairness_section('a'),
                    explainability_section('a'),
                    robustness_section('a'),
                    methodology_section('a'),
                    dcc.Store(id='training_data', storage_type='session'),
                    dcc.Store(id='test_data', storage_type='session')
                ], id="analysis_section")
            ],
                width=12, 
                className="mt-2 pt-2 pb-2 mb-2",
                style={
                    "border": "1px solid #d8d8d8",
                    "borderRadius": "6px"
                }   
            ),
        ], no_gutters=False)
    ])
])

