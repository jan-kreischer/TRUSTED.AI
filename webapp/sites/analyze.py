import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px
from app import app
import pandas as pd
import os
import json
import glob
import shutil
from helpers import *
from config import SCENARIOS_FOLDER_PATH, FAIRNESS_COLOR, EXPLAINABILITY_COLOR, ROBUSTNESS_COLOR, METHODOLOGY_COLOR, TRUST_COLOR
from sites.algorithm.helper_functions import get_performance_table, get_final_score, get_case_inputs, trusting_AI_scores, get_trust_score
from pillars.fairness.class_balance import compute_class_balance
import dash_table
import numpy as np
from sites import config_panel
import plotly.express as px
import plotly.graph_objects as go


config_fairness, config_explainability, config_robustness, config_methodology, config_pillars = 0, 0, 0 ,0,0
for config in ["config_pillars","config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open("sites/algorithm/"+config+".json") as file:
            exec("%s = json.load(file)" % config)

pillars = ['fairness', 'explainability', 'robustness', 'methodology']
main_config = dict(fairness=config_fairness, explainability=config_explainability, 
                   robustness=config_robustness, methodology=config_methodology, pillars=config_pillars)

with open('configs/default.json', 'w') as outfile:
                json.dump(main_config, outfile, indent=4)

# === SECTIONS ===

def general_information_section():
    return html.Div([
                                        dbc.Button(
            html.I(className="fas fa-backspace"),
            id="delete_solution_button", 
            n_clicks=0,
            style={"float": "right"}
        ),
daq.BooleanSwitch(id='toggle_charts',
                on=False,
                label='Alternative Style',
                labelPosition="top",
                color = TRUST_COLOR,   
                style={"float": "right"}
            ),
                    html.Div([html.H2("• General")]),
                    html.Div([], id="general_description"),
                    html.Div([], id="performance_div"),
                    ])


def trust_section(c):
    return html.Div([ 
        html.Div([
            html.Hr(),
            html.Div([daq.BooleanSwitch(id='show_weighting',
                      on=False,
                      label='Show Weighting',
                      labelPosition="top",
                      color = TRUST_COLOR,
                      style={"float": "right"}
                     
                    )]),
            html.H2("• Trustworthiness"),
        ], id="trust_section_heading", className="mt-2 mb-4"),

        html.Div([], id="trust_overview"),
        html.H3("Overall Score", className="text-center"),
        html.Div([], id="trust_star_rating", className="star_rating, text-center"),
        dcc.Graph(id='spider', style={'display': 'none'}),
        dcc.Graph(id='bar', style={'display': 'block'}),
        html.Div([], id="trust_details"),
        html.Hr()
    ], id="trust_section", style={"display": "None"})

def pillar_section(pillar):
        return html.Div([
                html.Div([
                    dbc.Button(
                        html.I(className="fas fa-chevron-down"),
                        id="toggle_{}_details".format(pillar),
                        className="mb-3",
                        n_clicks=0,
                        style={"float": "right"}
                    ),
                    daq.BooleanSwitch(id='show_{}_mappings'.format(pillar),
                      on=False,
                      label='Show Mappings',
                      labelPosition="top",
                      color = TRUST_COLOR,
                      style={"float": "right"}
                    ),
                    html.H2("• {}".format(pillar.upper()), className="mb-5"),
                ], id="{}_section_heading".format(pillar.lower())),
                    html.Div([], id="{}_overview".format(pillar)),
                    html.H3("{0}-Score".format(pillar), className="text-center"),
                    html.Div([], id="{}_star_rating".format(pillar), className="star_rating, text-center"),
                    dcc.Graph(id='{}_spider'.format(pillar), style={'display': 'none'}),
                    dcc.Graph(id='{}_bar'.format(pillar), style={'display': 'block'}),    
                    dbc.Collapse(
                        html.P("{} Details".format(pillar)),
                        id="{}_details".format(pillar),
                        is_open=False,
                    ),
                    html.Hr(style={"size": "10"}),

                ], id="{}_section".format(pillar), style={"display": "None"})
    

def alert_section(name):
    return html.Div([], id="{}_alert_section".format(name), className="text-center", style={"color":"Red"})


SECTIONS = ['trust', 'fairness', 'explainability', 'robustness', 'methodology']
for s in SECTIONS:
    @app.callback(
        [Output("{0}_details".format(s), "is_open"),
        Output("{0}_details".format(s), "style")],
        [Input("toggle_{0}_details".format(s), "n_clicks")],
        [State("{0}_details".format(s), "is_open")],
        prevent_initial_call=True
    )
    def toggle_detail_section(n, is_open):
        #app.logger.info("toggle {0} detail section".format(s))
        if is_open:
            return (not is_open, {'display': 'None'})
        else:
            return (not is_open, {'display': 'Block'})

layout = html.Div([
    config_panel.layout,
    dbc.Container([
     
        dbc.Row([
            dcc.Store(id='result'),
            dbc.Col([html.H1("Analyze", className="text-center")], width=12, className="mb-2 mt-1"),
            
            dbc.Col([dcc.Dropdown(
                    id='solution_set_dropdown',
                    options= get_solution_sets(),
                    value=None,

                    placeholder='Select Solution'
                )], width=12, style={"marginLeft": "0 px", "marginRight": "0 px"}, className="mb-1 mt-1"
            ),
                
            dbc.Col([
                dcc.ConfirmDialog(
                    id='delete_solution_confirm',
                    message='Are you sure that you want to delete this solution?',
                ),
                html.Div([], id="delete_solution_alert"),
                html.Div([], id="analyze_alert_section"),
                
                html.Div([
                    general_information_section(),
                    trust_section('a'),
                    pillar_section("fairness"),
                    pillar_section("explainability"),
                    pillar_section("robustness"),
                    pillar_section("methodology"),
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

@app.callback(
    Output(component_id="bar", component_property='style'),
    Output(component_id="fairness_bar", component_property='style'),
    Output(component_id="explainability_bar", component_property='style'),
    Output(component_id="robustness_bar", component_property='style'),
    Output(component_id="methodology_bar", component_property='style'),
    Output(component_id="spider", component_property='style'),
    Output(component_id="fairness_spider", component_property='style'),
    Output(component_id="explainability_spider", component_property='style'),
    Output(component_id="robustness_spider", component_property='style'),
    Output(component_id="methodology_spider", component_property='style'),
    Input('toggle_charts', 'on')
)
def toggle_charts(visibility_state):
    if visibility_state == True:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    if visibility_state == False:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

@app.callback(Output('solution_set_dropdown', 'options'),
              Input('solution_set_dropdown', 'nclicks'))
def update_solution_set_dropdown(n_clicks):
    return get_solution_sets()
    
@app.callback(Output('general_description', 'children'),
              Input('solution_set_dropdown', 'value'), prevent_initial_call=True)
def show_general_description(solution_set_path):
    factsheet = read_factsheet(solution_set_path)
    description = ""
    if "general" in factsheet and "description" in factsheet["general"]:
        description = factsheet["general"]["description"]
    return [html.H4("Description: "), description]
    
@app.callback([Output('solution_set_dropdown', 'value'),
              Output('delete_solution_alert', 'children')],
              [Input('delete_solution_confirm', 'submit_n_clicks'),
               Input('uploaded_solution_set_path', 'data')],
              State('solution_set_dropdown', 'value'))
def update_output(submit_n_clicks, uploaded_solution_set, solution_set_path):
    
    if not solution_set_path:
        return "",[]
    
    print("UPDATE OUTPUT {}".format(submit_n_clicks))
    if submit_n_clicks:
        app.logger.info("Deletign {}".format(solution_set_path))
        try:
            shutil.rmtree(solution_set_path, ignore_errors=False)
        except Exception as e:
            print(e)
            raise
        return "", html.H3("Deleted solution", className="text-center", style={"color": "Red"})
    else:
        if uploaded_solution_set:
            return uploaded_solution_set["path"], []
        
# === TRUST ===
# @app.callback(
#     [Output("trust_overview", 'children'),
#     Output("trust_details", 'children')],
#     [Input('solution_set_dropdown', 'value'), State("result","data")], prevent_initial_call=True)
# def analyze_trust(solution_set_path, result, config):
#     if solution_set_path and result:
        
#         dcc.Graph(id='spider-old', style={'display': 'none'})
        
#         return 
#     else:
#         return [], []


# === FAIRNESS ===
fairness_metrics = ["class_balance", "statistical_parity_difference", "equal_opportunity_difference", "average_odds_difference", "disparate_impact", "theil_index", "euclidean_distance", "mahalanobis_distance", "manhattan_distance"]


for m in fairness_metrics:
    @app.callback(
        [Output("{0}_details".format(m), "is_open"),
        Output("{0}_details".format(m), "style")],
        [Input("toggle_{0}_details".format(m), "n_clicks")],
        [State("{0}_details".format(m), "is_open")],
        prevent_initial_call=True
    )
    def toggle_detail_section(n, is_open):
        #app.logger.info("toggle {0} detail section".format(s))
        if is_open:
            return (not is_open, {'display': 'None'})
        else:
            return (not is_open, {'display': 'Block'})

@app.callback(
    [Output("fairness_overview", 'children'),
    Output("fairness_details", 'children')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def analyze_fairness(solution_set_path):
    if solution_set_path == "":
        return ["", ""]
    print("Analyze Fairness {}".format(solution_set_path))
    if solution_set_path is not None:
        train_data =  read_train(solution_set_path)
        test_data =  read_test(solution_set_path)

        features = list(train_data.columns)
        target_column=""
        factsheet = None

        factsheet_path = os.path.join(solution_set_path,"factsheet.json") 
        # Check if a factsheet.json file already exists in the target directory
        if os.path.isfile(factsheet_path):

            f = open(factsheet_path,)
            factsheet = json.load(f)
            
            target_column = ""
            if "general" in factsheet and "target_column" in factsheet["general"]:
                target_column = factsheet["general"]["target_column"]
            print("target_column {}".format(target_column))
            
            protected_feature = ""
            if "fairness" in factsheet and "protected_feature" in factsheet["fairness"]:
                protected_feature = factsheet["fairness"]["protected_feature"]
            print("protected_feature {}".format(protected_feature))
            
            privileged_class_definition = ""
            if "fairness" in factsheet and "privileged_class_definition" in factsheet["fairness"]:
                privileged_class_definition = factsheet["fairness"]["privileged_class_definition"]
            print("privileged_class_definition {}".format(privileged_class_definition))
            
            f.close()
        # Create a factsheet
        else:
            print("No factsheet exists yet")


        solution_set_label_select_options = list(map(lambda x: {"label": x, "value": x}, features))
        solution_set_label_select = html.Div([
            "Select Target Column", 
            dcc.Dropdown(
                id="solution_set_label_select",
                options=solution_set_label_select_options,
                value=target_column
            ),
        ])
        
        protected_feature_select_options = list(map(lambda x: {"label": x, "value": x}, features))
        protected_feature_select = html.Div([
            "Select Protected Feature", 
            dcc.Dropdown(
                id="protected_feature_select",
                options=protected_feature_select_options,
                value=protected_feature
            ),
        ])
        
        privileged_class_definition = html.Div([
            "Define Privileged Class",
            html.Br(),
            dcc.Input(
                id="privileged_class_definition",
                type="text",
                placeholder="e.g lambda x: x >= 25",
                style={'width': '100%'}
            ),
        ])

        sections = [html.Hr(), html.H3("▶ Fairness Configuration"), solution_set_label_select, protected_feature_select, privileged_class_definition, html.Hr(), html.H3("▶ Fairness Metrics")]
        

        for i in range(len(fairness_metrics)):
            metric = fairness_metrics[i]
            metric_id = metric
            metric_name = metric_id.replace("_", " ")
            sections.append(html.Div([dbc.Button(
                        html.I(className="fas fa-chevron-down"),
                        id="toggle_{}_details".format(metric),
                        className="mb-3",
                        n_clicks=0,
                        style={"float": "right"}
                    ),html.H4("1.{0} {1}".format(i+1, metric_name)), 
                    dbc.Collapse(
                        html.Div("{}_details".format(metric)),
                        id="{}_details".format(metric),
                        is_open=False,          
                    ),
            ], id="{}_section".format(metric), className="mb-5 mt-5"))
        
        
        return [], sections
    else:
        return [], [html.H1("Nothing")]

def update_factsheet(factsheet_path, key, value):
    print("update factsheet {0} with {1}  {2}".format(factsheet_path, key, value))
    jsonFile = open(factsheet_path, "r") # Open the JSON file for reading
    data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    ## Working with buffered content
    data[key] = value

    ## Save our changes to JSON file
    jsonFile = open(factsheet_path, "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()
     
'''
The following function updates
'''
@app.callback(
    Output("class_balance_details", 'children'),
    [Input("solution_set_label_select", 'value'), 
    State('training_data', 'data'),
    State("solution_set_dropdown", 'value')])
def fairness_metrics_class_balance(label, jsonified_training_data, solution_set_path):
    training_data = read_train(solution_set_path)
    graph = dcc.Graph(figure=px.histogram(training_data, x=label, opacity=1, title="Label vs Label Occurence", color_discrete_sequence=[FAIRNESS_COLOR]))
    #compute_class_balance("hi")
    update_factsheet(r"{}/factsheet.json".format(solution_set_path), "target_column", label)
    return [graph]
    
    #print("label {}".format(label))
    #print("JSONIFIED TRAINING DATA {}".format(jsonified_training_data))
    #training_data = pd.read_json(jsonified_training_data, orient='split')
    #print("Training data")
    #print(dff.head(5))
    #figure = create_figure(dff)
    #fig = px.histogram(train_data, x=label)
    #graph = dcc.Graph(figure=px.histogram(training_data, x=label))
    #return [html.H3("Selected {} as label column. Computing class balance now.".format(label)), graph]
    
'''
The following function updates
'''
@app.callback(
    Output("statistical_parity_difference_details", 'children'),
    [Input("solution_set_label_select", 'value'), 
    State('training_data', 'data'),
    State("solution_set_dropdown", 'value')])
def fairness_metrics_statistical_parity_difference(label, jsonified_training_data, solution_set_path):
    training_data = read_train(solution_set_path)
    graph = dcc.Graph(figure=px.histogram(training_data, x=label, opacity=0.5, title="Label vs Label Occurence", color_discrete_sequence=['#00FF00']))
    #compute_class_balance("hi")
    update_factsheet(r"{}/factsheet.json".format(solution_set_path), "target_column", label)
    return [graph]
    
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
    if solution_set_path != "":
        training_data = read_train(solution_set_path)
        test_data = read_test(solution_set_path)
        compute_train_test_split(solution_set_path)
        return training_data.to_json(date_format='iso', orient='split'), test_data.to_json(date_format='iso', orient='split'), 
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
# @app.callback(
#     [Output("explainability_overview", 'children'),
#     Output("explainability_details", 'children')],
#     [Input("solution_set_dropdown", 'value')], prevent_initial_call=True)
# def analyze_explainability(solution_set_path):
#     if solution_set_path is not None:

#         df = pd.DataFrame(dict(
#             r=[1, 5, 2, 2, 3],
#             theta=['processing cost','mechanical properties','chemical stability',
#            'thermal stability', 'device integration']))
#         fig = px.line_polar(df, r='r', theta='theta', line_close=True)
#         fig.update_traces(fill='toself')
#         explainability_overview = dcc.Graph(figure=fig)
#         return [explainability_overview], []
#     else:
#         return [], []
        

# === ROBUSTNESS ===
# @app.callback(
#     [Output("robustness_overview", 'children'),
#     Output("robustness_details", 'children')],
#     [Input("solution_set_dropdown", 'value')], prevent_initial_call=True)
# def analyze_robustness(solution_set_path):
#     if solution_set_path is not None:
#         df = pd.DataFrame(dict(
#             r=[1, 5, 2, 2, 3],
#             theta=['processing cost','mechanical properties','chemical stability',
#            'thermal stability', 'device integration']))
#         fig = px.line_polar(df, r='r', theta='theta', line_close=True)
#         fig.update_traces(fill='toself')
#         explainability_overview = dcc.Graph(figure=fig)
#         return [explainability_overview], [html.Div("Robustness Metrics")]
#     else:
#         return [], []
 
# === METHODOLOGY ===
# @app.callback(
#     [Output("methodology_overview", 'children'),
#     Output("methodology_details", 'children')],
#     [Input("solution_set_dropdown", 'value')], prevent_initial_call=True)
# def analyze_methodology(solution_set_path):
#     if solution_set_path is not None:
#         df = pd.DataFrame(dict(
#             r=[1, 5, 2, 2, 3],
#             theta=['processing cost','mechanical properties','chemical stability',
#            'thermal stability', 'device integration']))
#         fig = px.line_polar(df, r='r', theta='theta', line_close=True)
#         fig.update_traces(fill='toself')
#         explainability_overview = dcc.Graph(figure=fig)
#         return [explainability_overview], [html.Div("Methodology Metrics a", style={"display": "None"})]
#     else:
#         return [], []
   
@app.callback(
    Output(component_id="trust_section", component_property='style'),
    Output(component_id="fairness_section", component_property='style'),
    Output(component_id="explainability_section", component_property='style'),
    Output(component_id="robustness_section", component_property='style'),
    Output(component_id="methodology_section", component_property='style'),
    [Input("bar", 'figure')], prevent_initial_call=True)
def toggle_pillar_section_visibility(path):
    if path is not None:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
   
@app.callback(
    [Output('analyze_alert_section', 'children'),
    Output('analysis_section', 'style')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def analyze_methdology(solution_set_path):
    button = []
    alerts = []
    style={'display': 'block'}
    if solution_set_path is not None:
        button = dbc.Button(
            html.I(className="fas fa-backspace"),
            id="delete_button", 
            n_clicks=0,
            style={"float": "right"}
        ),
        
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

@app.callback(Output('delete_solution_confirm', 'displayed'),
              Input('delete_solution_button', 'n_clicks'), prevent_initial_call=True)
def display_confirm(n_clicks):
    if n_clicks:
        return True
    else:
        return False
    
# #data_stores = html.Div[dcc.Store(id='test_data'),  dcc.Store(id='train_data'),  dcc.Store(id='model'),  dcc.Store(id='factsheet')]
@app.callback(Output('performance_div', 'children'), 
          Input('solution_set_dropdown', 'value'))
def get_performance(solution_set_dropdown):
    
    if not solution_set_dropdown:
        return html.P()
    
    test, train, model, factsheet = read_scenario(solution_set_dropdown)
    
    target_column = factsheet["general"].get("target_column")
    
    performance =  get_performance_table(model, test, target_column).transpose()
    
    performance_table = dash_table.DataTable(
                            id='table',
                            columns=[{"name": i, "id": i} for i in performance.columns],
                            data=performance.to_dict('records'),
                            style_table={"table-layout": "fixed", "width": "100%"}

    )
    
    performance_div = html.Div([html.H5("Performance metrics", style={"width": "100%","text-align": "center", "margin-right": "auto", "margin-left": "auto" }),
                                performance_table],style={"width": "100%"})
        
    return performance_div

@app.callback(Output('result', 'data'), 
          [Input('solution_set_dropdown', 'value'),
          Input("input-config","data")])
def store_result(solution_set_dropdown, config):
    
        if not solution_set_dropdown:
            return None
    
        if not config:
            with open('configs/default.json','r') as f:
                main_config = json.loads(f.read())
        else:
            main_config = json.loads(config)
            
        test, train, model, factsheet = read_scenario(solution_set_dropdown)
    
        final_score, results, properties = get_final_score(model, train, test, main_config, factsheet)
        trust_score = get_trust_score(final_score, main_config["pillars"])
        
        def convert(o):
            if isinstance(o, np.int64): return int(o)  
           
            
        data = {"final_score":final_score,
                "results":results,
                "trust_score":trust_score,
                "properties" : properties}
        
        return json.dumps(data,default=convert)
    
@app.callback(
      [Output('bar', 'figure'),
       Output('spider', 'figure'),
       Output('fairness_bar', 'figure'),
       Output('explainability_bar', 'figure'),
       Output('robustness_bar', 'figure'),
       Output('methodology_bar', 'figure'),
       Output('fairness_spider', 'figure'),
       Output('explainability_spider', 'figure'),
       Output('robustness_spider', 'figure'),
       Output('methodology_spider', 'figure'),
       Output('trust_star_rating', 'children'),
       Output('fairness_star_rating', 'children'),
       Output('explainability_star_rating', 'children'),
       Output('robustness_star_rating', 'children'),
       Output('methodology_star_rating', 'children'),
       Output("robustness_details", 'children')],
      [Input('result', 'data'),Input("hidden-trigger", "value")])  
def update_figure(data, trig):
     
      # if not config or trig != "apply-config.n_clicks":
      #     with open('configs/default.json','r') as f:
      #         main_config = json.loads(f.read())
      # else:
      #     main_config = json.loads(config)
         
      # np.random.seed(6)
      if data is None:
          return [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "", "", "", "", "", ""]
      result = json.loads(data)
      final_score, results, properties = result["final_score"] , result["results"], result["properties"]
      trust_score = result["trust_score"]
      pillars = list(final_score.keys())
      values = list(final_score.values()) 
        
      colors = [FAIRNESS_COLOR, EXPLAINABILITY_COLOR, ROBUSTNESS_COLOR, METHODOLOGY_COLOR]
      
      # barchart
      chart_list=[]
      bar_chart = go.Figure(data=[go.Bar(
          x=pillars,
          y=values,
          marker_color=colors
              )])
      bar_chart.update_layout(title_text='<b style="font-size: 48px;">{}/5</b>'.format(trust_score), title_x=0.5)
      chart_list.append(bar_chart)
     
      #spider
      spider_plt = px.line_polar(r=values, theta=pillars, line_close=True, title='<b style="font-size:42px;">{}/5</b>'.format(trust_score))
      spider_plt.update_layout(title_x=0.5)
      spider_plt.update_traces(fill='toself', fillcolor=TRUST_COLOR, marker_color=TRUST_COLOR, marker_line_color=TRUST_COLOR, marker_line_width=0, opacity=0.6)
      chart_list.append(spider_plt)
     
      #barcharts
      for n, (pillar , sub_scores) in enumerate(results.items()):
          title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
          categories = list(map(lambda x: x.replace("_",' '), sub_scores.keys()))
          values = list(map(float, sub_scores.values()))
          if np.isnan(values).any():
              nonNanCategories = list()
              nonNanValues = list()
              for c, v in zip(categories, values):
                  if not np.isnan(v):
                      nonNanCategories.append(c)
                      nonNanValues.append(v)
              categories = nonNanCategories
              values = nonNanValues
          bar_chart_pillar = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors[n])])
          bar_chart_pillar.update_layout(title_text=title, title_x=0.5)
          chart_list.append(bar_chart_pillar)
         
      #spider charts
      for n, (pillar , sub_scores) in enumerate(results.items()):
          title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
          categories = list(map(lambda x: x.replace("_",' '), sub_scores.keys()))
          val = list(map(float, sub_scores.values()))
          if np.isnan(values).any():
              nonNanCategories = list()
              nonNanValues = list()
              for c, v in zip(categories, values):
                  if not np.isnan(v):
                      nonNanCategories.append(c)
                      nonNanValues.append(v)
              categories = nonNanCategories
              val = nonNanValues
          spider_plt_pillar = px.line_polar(r=val, theta=categories, line_close=True, title=title)
          spider_plt_pillar.update_traces(fill='toself', fillcolor=colors[n], marker_color=colors[n],marker_line_width=1.5, opacity=0.6)
          spider_plt_pillar.update_layout(title_x=0.5)
          chart_list.append(spider_plt_pillar)

      robutness_properties = properties["robustness"]
      robustness_detail = []
      for n, (metric , metric_properties) in enumerate(robutness_properties.items()):
          categories = list(map(lambda x: x.replace("_",' '), metric_properties.keys()))
          val = list(map(float, metric_properties.values()))
          robustness_detail.append(html.Div(html.H4(metric),className="mb-5 mt-5"))

          for cat, v in zip(categories, val):
              robustness_detail.append(html.Div(html.H5(cat), className="mb-5 mt-5"))
              robustness_detail.append(html.Div(html.H5(v), className="mb-5 mt-5"))

      robustness_detail2 = [dbc.Col(children=robustness_detail)]

      star_ratings = []
      star_ratings.append(show_star_rating(trust_score))
      star_ratings.append(show_star_rating(final_score["fairness"]))
      star_ratings.append(show_star_rating(final_score["explainability"]))
      star_ratings.append(show_star_rating(final_score["robustness"]))
      star_ratings.append(show_star_rating(final_score["methodology"]))
      return chart_list + star_ratings + robustness_detail2
 
config_panel.get_callbacks(app)

