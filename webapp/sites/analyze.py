# === IMPORTS ===
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
from config import *
from algorithms.trustworthiness import trusting_AI_scores, get_trust_score, get_final_score
import dash_table
import numpy as np
from sites import config_panel
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# === CONFIG ===

config_fairness, config_explainability, config_robustness, config_methodology, config_pillars = 0, 0, 0 ,0,0
for config in ["config_pillars","config_fairness", "config_explainability", "config_robustness", "config_methodology"]:
    with open(os.path.join(METRICS_CONFIG_PATH, config + ".json")) as file:
            exec("%s = json.load(file)" % config)

pillars = ['fairness', 'explainability', 'robustness', 'methodology']
weight_config = dict(fairness=config_fairness["weights"], explainability=config_explainability["weights"], 
                   robustness=config_robustness["weights"], methodology=config_methodology["weights"], pillars=config_pillars)

mappings_config = dict(fairness=config_fairness["parameters"], explainability=config_explainability["parameters"], 
                   robustness=config_robustness["parameters"], methodology=config_methodology["parameters"])

with open('configs/weights/default.json', 'w') as outfile:
                json.dump(weight_config, outfile, indent=4)

with open('configs/mappings/default.json', 'w') as outfile:
                json.dump(mappings_config, outfile, indent=4)

for s in SECTIONS[1:]:
    with open('configs/mappings/{}/default.json'.format(s), 'w') as outfile:
                json.dump(mappings_config[s], outfile, indent=4)

# === METRICS ===
fairness_metrics = FAIRNESS_METRICS
explainability_metrics = EXPLAINABILITY_METRICS
robustness_metrics = ROBUSTNESS_METRICS
methodology_metrics = METHODOLOGY_METRICS

# === SECTIONS ===
def general_section():
    return html.Div([
        dbc.Button(
            html.I(className="fas fa-backspace"),
            id="delete_solution_button", 
            n_clicks=0,
            style={"float": "right", "backgroundColor": SECONDARY_COLOR}
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
                html.Div(["Performance Metrics Section"], id="performance_metrics_section"),
                dcc.Store(id='input-mappings'),
                ])

def trust_section():
    return html.Div([ 
        html.Div([
            html.Hr(),
            html.Div([daq.BooleanSwitch(id='show_weighting',
                      on=False,
                      label='Show Weights',
                      labelPosition="top",
                      color = TRUST_COLOR,
                      style={"float": "right"}
                     
                    )]),
            html.H2("• Trustworthiness"),
        ], id="trust_section_heading", className="mt-2 mb-5"),

        html.Div([], id="trust_overview"),
        html.H3("Overall Score", className="text-center mt-2"),
        html.Div([], id="trust_star_rating", className="star_rating, text-center"),
        html.B(["X/5"], id="trust_score", className="text-center", style={"display": "block","font-size":"32px"}),
        dcc.Graph(id='spider', style={'display': 'none'}),
        dcc.Graph(id='bar', style={'display': 'block'}),
        html.Div([], id="trust_details"),
        html.Hr()
    ], id="trust_section", style={"display": "None"})

def alert_section(name):
    return html.Div([], id="{}_alert_section".format(name), className="text-center", style={"color":"Red"})

for pillar in SECTIONS[1:]:
    @app.callback(
        Output("modal-{}-mapping".format(pillar), "is_open"),
        [Input('save-mapping-{}'.format(pillar), "n_clicks"),Input("save-{}-mapping".format(pillar), "n_clicks")],
        State("modal-{}-mapping".format(pillar), "is_open"))
    def update_output(n, n2, is_open):
        if n or n2:
            return not is_open
        else:
            return is_open
        
    @app.callback(
            Output("mapping-dropdown-{}".format(pillar), "options"),
            Input("modal-saved-{}".format(pillar), "is_open"),
            State("mapping-dropdown-{}".format(pillar), "className"))
    def update_options(n, pillar):
        options = list(map(lambda name:{'label': name[:-5], 'value': "configs/mappings/{}/{}".format(pillar,name)} ,os.listdir("configs/mappings/{}".format(pillar))))
        return options
    
    @app.callback(
                 list(map(lambda i: Output(i, "value"),mapping_panel(pillar)[1])), 
                Input("mapping-dropdown-{}".format(pillar), 'value'),
                State("mapping-dropdown-{}".format(pillar), "className"))
    def update_mapping_input(conf_name,pillar):
        
        with open(conf_name ,'r') as f:
                config = json.loads(f.read())
                
        output = []
        ids = mapping_panel(pillar)[1]
        for i in ids:
            metric, param = i.split("-")
            output.append(str(config[metric][param]["value"]))
            
        return output
    
    @app.callback(
        [Output("modal-saved-{}".format(pillar), "is_open"),Output("mapping-dropdown-{}".format(pillar), "value")],
        [Input("save-{}-mapping".format(pillar), "n_clicks")],
        [State("modal-saved-{}".format(pillar), "is_open"),State("mapping-name-{}".format(pillar), "value"), State(mapping_panel(pillar)[1][0],"name")]+ 
        list(map(lambda i: State(i, "value"),mapping_panel(pillar)[1])))
    def save_mapping(n, is_open, conf_name, pillar, *args):
        
        if conf_name:  
            inputs= dict()
        
            for name, val in zip(mapping_panel(pillar)[1] , args):
                try:
                    res = eval(val)
                except (SyntaxError, NameError, TypeError, ZeroDivisionError):
                    res = val
               
                inputs[name] = res
            
            with open('configs/mappings/{}/default.json'.format(pillar),'r') as f:
                config_file = json.loads(f.read())
                
            for i in mapping_panel(pillar)[1]:
                 metric, param = i.split("-")
                 config_file[metric][param]["value"] = inputs[i]
            with open('configs/mappings/{}/{}.json'.format(pillar,conf_name), 'w') as outfile:
                json.dump(config_file, outfile, indent=4)
                    
            return not is_open, 'configs/mappings/{}/{}.json'.format(pillar,conf_name)
        else:
            return is_open, 'configs/mappings/{}/default.json'.format(pillar)

for s in SECTIONS:
    @app.callback(
        [Output("{0}_mapping".format(s), "is_open"),
        Output("{0}_mapping".format(s), "style")],
        [Input("toggle_{}_mapping".format(s), "on")],
        [State("{0}_mapping".format(s), "is_open")],
        prevent_initial_call=True
    )
    def toggle_mapping_section(n, is_open):
        #app.logger.info("toggle {0} detail section".format(s))
        if is_open:
            return (not is_open, {'display': 'None'})
        else:
            return (not is_open,  {"background-color": "rgba(255,228,181,0.5)",'padding-bottom': 20, 'display': 'Block'})

for s in SECTIONS[1:]:
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

#@app.callback(Output('solution_set_dropdown', 'options'),
#              Input('solution_set_dropdown', 'nclicks'))
#def update_solution_set_dropdown(n_clicks):
#    return get_solution_sets()


@app.callback(Output('input-mappings', 'data'), 
        list(map(lambda i: Input('apply-mapping-{}'.format(i), "n_clicks"),SECTIONS[1:])),
        list(map(lambda i: State(i, "value"), sum(list(map(lambda s: mapping_panel(s)[1],SECTIONS[1:])),[]) )))
       
def store_mappings_config(n1, n2, n3, n4, *args):
    
    inputs= dict()
    
    for name, val in zip(sum(list(map(lambda s: mapping_panel(s)[1],SECTIONS[1:])),[]) , args):
        try:
               res = eval(val)
        except (SyntaxError, NameError, TypeError, ZeroDivisionError):
           res = val
           
        inputs[name] = res
        
    with open('configs/mappings/default.json','r') as f:
            config_file = json.loads(f.read())
    
    
    pillars = SECTIONS[1:]
    ids = list(map(lambda s: mapping_panel(s)[1],SECTIONS[1:]))
    for pillar, map_ids in zip(pillars, ids):
        for i in map_ids:
            metric, param = i.split("-")
            config_file[pillar][metric][param]["value"] = inputs[i]
    
    return json.dumps(config_file)
    
@app.callback(Output('general_description', 'children'),
              Input('solution_set_dropdown', 'value'), prevent_initial_call=True)
def show_general_description(solution_set_path):
    if not solution_set_path:
        return ""
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
    
# === FAIRNESS ===
@app.callback(
    Output("fairness_configuration", 'children'),
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def fairness_configuration(solution_set_path):
    if solution_set_path == "":
        return []
    if solution_set_path is not None:
        train_data =  read_train(solution_set_path)
        test_data =  read_test(solution_set_path)

        features = list(train_data.columns)
        target_column=""
        factsheet = None

        factsheet_path = os.path.join(solution_set_path, "factsheet.json") 
        # Check if a factsheet.json file already exists in the target directory



        if os.path.isfile(factsheet_path):

            f = open(factsheet_path,)
            factsheet = json.load(f)
                   
            target_column = ""
            if "general" in factsheet and "target_column" in factsheet["general"]:
                target_column = factsheet["general"]["target_column"]
            
            protected_feature = ""
            if "fairness" in factsheet and "protected_feature" in factsheet["fairness"]:
                protected_feature = factsheet["fairness"]["protected_feature"]
            print("protected_feature: {}".format(protected_feature))
            
            protected_group = ""
            if "fairness" in factsheet and "protected_group" in factsheet["fairness"]:
                protected_group = factsheet["fairness"]["protected_group"]
            
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
        
        protected_group_definition = html.Div([
            "Define Protected Group",
            html.Br(),
            dcc.Input(
                id="protected_group_definition",
                type="text",
                placeholder="e.g lambda x: x[protected_feature] < 25",
                value=protected_group,
                style={'width': '100%'}
            ),
        ])


        sections = [html.Hr(), html.H3("▶ Fairness Configuration"), solution_set_label_select, protected_feature_select, protected_group_definition, html.Hr()]
        
        #for i in range(len(fairness_metrics)):
        #    metric_id = fairness_metrics[i]
        #    sections.append(create_metric_details_section(metric_id, i))
        return sections
    else:
        return []

# === EXPLAINABILITY ===
@app.callback(
    Output("explainability_details", 'children'),
    Input('result', 'data'), prevent_initial_call=True)
def explainability_details(data):
    if not data:
        return []
    result = json.loads(data)
    properties = result["properties"]
    metrics = list(properties["explainability"].keys())
    
    sections = [html.H3("▶ Explainability Metrics")]
    for i in range(len(metrics)):
            metric_id = metrics[i]
            score = result["results"]["explainability"][metric_id]
            sections.append(create_metric_details_section(metric_id, i, 2, True,score))
    return sections

@app.callback(
    list(map(lambda o: Output("{}_details".format(o), 'children'), explainability_metrics)),
    Input('result', 'data'), prevent_initial_call=False)    
def metric_detail(data):
  if data is None:
      return [], [], [], []
  else:
      output = []
      result = json.loads(data)
      properties = result["properties"]
      for metric in explainability_metrics:
          metric_properties = properties["explainability"][metric]
          if not metric_properties:
              output.append(html.Div())
          else:
              prop = []
              for k, p in metric_properties.items():
           
                  if k == "importance" :
                        importance = p[1]
                        pct_dist = metric_properties["pct_dist"][1]
                        pct_mark = importance["labels"][int(float(pct_dist[:-1])/100 * len(importance["value"]))-1]
                        
                        fig = go.Figure(data=go.Scatter(x=importance["labels"][:30], y=importance["value"][:30],mode='lines+markers'))
                        fig.add_vline(x=pct_mark, line_width=3, line_dash="dash")
                        fig.add_trace(go.Scatter(
                            x=[importance["labels"][int(float(pct_dist[:-1])/100 * len(importance["value"]))-3]], y=[max(importance["value"])],
                            text="60% Threshold",
                            mode="text",
                        ))
                        fig.update_layout(showlegend=False)    
                        prop.append(html.Div(["Features importance of top 30 features:",
                            dcc.Graph(id='test',figure=fig, style={'display': 'block'})]))
                  else:
                      prop.append(html.Div("{} : {}".format(p[0], p[1])))
                  prop.append(html.Br())
              output.append(html.Div(prop))
      return output
     
'''
The following function updates
'''
@app.callback(
    [Output("class_balance_details", 'children')],
    [Input("solution_set_label_select", 'value'), 
    State('training_data', 'data'),
    State("solution_set_dropdown", 'value')], prevent_initial_call=True)
def class_balance(label, jsonified_training_data, solution_set_path):
    training_data = read_train(solution_set_path)
    graph = dcc.Graph(figure=px.histogram(training_data, x=label, opacity=1, title="Label vs Label Occurence", color_discrete_sequence=[FAIRNESS_COLOR]))
    return [graph]
       
'''
The following function updates
'''
@app.callback(
    [Output("statistical_parity_difference_details", 'children'), Output("statistical_parity_difference_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def statistical_parity_difference(data):
    if data is None:
        return [NO_DETAILS], [NO_SCORE_FULL]
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["fairness"]["statistical_parity_difference"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["fairness"]["statistical_parity_difference"]))
    
    
#@app.callback(
#    [Output("f1_score_details", 'children'), Output("f1_score_score", 'children')],
#    [Input('result', 'data')])
#def f1_score(data):
#    if data is None:
#        return [], []
#    else:
#        result = json.loads(data)
#        properties = result["properties"]
#        metric_properties = properties["methodology"]["f1_score"]
#        metric_scores = result["results"]
#        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["f1_score"]))
    
@app.callback(
    [Output('training_data', 'data'),
     Output('test_data', 'data')],
    [Input('solution_set_dropdown', 'value')], prevent_initial_call=True)
def load_data(solution_set_path):
    if solution_set_path != "":
        training_data = read_train(solution_set_path)
        test_data = read_test(solution_set_path)
        #compute_train_test_split(solution_set_path)
        return training_data.to_json(date_format='iso', orient='split'), test_data.to_json(date_format='iso', orient='split'), 
    else:
        return None, None

# === METHODOLOGY ===
@app.callback(
    [Output("trust_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def trust_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["trust_score"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]
    
@app.callback(
    [Output("methodology_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def methodology_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["methodology"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]
  
@app.callback(
    [Output("fairness_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def fairness_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["fairness"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]

@app.callback(
    [Output("robustness_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def robustness_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["robustness"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]
    
@app.callback(
    [Output("explainability_score", 'children')],
    [Input('result', 'data')], prevent_initial_call=True)
def explainability_score(analysis):
    if analysis:
        analysis = json.loads(analysis)
        score = analysis["final_score"]["explainability"]
        return ["{}/5".format(score)]
    else:
        return ["{}/5".format(NO_SCORE)]

@app.callback(
    [Output("f1_score_details", 'children'), Output("f1_score_score", 'children')],
    [Input('result', 'data')])
def f1_score(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["f1_score"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["f1_score"]))


# --- Normalization ---
@app.callback(
    [Output("normalization_details", 'children'), Output("normalization_score", 'children')],
    [Input('result', 'data')])
def normalization(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["normalization"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["normalization"]))

@app.callback(
    [Output("test_accuracy_details", 'children'), Output("test_accuracy_score", 'children')],
    [Input('result', 'data')])
def test_accuracy(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["test_accuracy"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["test_accuracy"]))

@app.callback(
    [Output("missing_data_details", 'children'), Output("missing_data_score", 'children')],
    [Input('result', 'data')])
def missing_data(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["missing_data"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["methodology"]["missing_data"]))


# --- Regularization ---
@app.callback(
    [Output("regularization_details", 'children'), Output("regularization_score", 'children')],
    [Input('result', 'data'),
     State('solution_set_dropdown', 'value')], prevent_initial_call=True)
def regularization(analysis, solution_set_path):
    if analysis and solution_set_path:
          analysis = json.loads(analysis)
          _, metric_scores, metric_properties = analysis["final_score"] , analysis["results"], analysis["properties"]
          metric_score = metric_scores["methodology"]["regularization"]
          regularization_technique = metric_properties["methodology"]["regularization"]["regularization_technique"]
          return html.Div("Regularization Technique: {}".format(regularization_technique)), html.H4("({}/5)".format(metric_score))
    else:
        return [], []
    
# --- Train Test Split ---
@app.callback(
    [Output("train_test_split_details", 'children'), Output("train_test_split_score", 'children')],
    [Input('result', 'data')])
def train_test_split(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["methodology"]["train_test_split"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["methodology"]["train_test_split"]))

# --- Factsheet Completeness ---
@app.callback(
    [Output("factsheet_completeness_details", 'children'), Output("factsheet_completeness_score", 'children')],
    [Input('result', 'data'),
     State('solution_set_dropdown', 'value')], prevent_initial_call=True)
def factsheet_completeness(analysis, solution_set_path):
      if analysis is None:
          return [], []
      else:
          analysis = json.loads(analysis)
          _, metric_scores, metric_properties = analysis["final_score"] , analysis["results"], analysis["properties"]
          metric_score = metric_scores["methodology"]["factsheet_completeness"]
          return html.Div("Train-Test-Split: {0}/{1}"), html.H4("({}/5)".format(metric_score))

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
def analyze_solution_completeness(solution_set_path):
    button = []
    alerts = []
    style={'display': 'block'}
    if solution_set_path is not None:
        button = dbc.Button(
            html.I(className="fas fa-backspace"),
            id="delete_button", 
            n_clicks=0,
            style={"float": "right", "backgroundColor": SECONDARY_COLOR}
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
    
@app.callback(Output('performance_metrics_section', 'children'), 
          Input('solution_set_dropdown', 'value'), prevent_initial_call=True)
def show_performance_metrics(solution_set_path):
    if not solution_set_path:
        return []
    else:
        test_data, training_data, model, factsheet = read_solution(solution_set_path)

        target_column = factsheet.get("general", {}).get("target_column", "")
        
        performance_metrics =  get_performance_metrics(model, test_data, target_column)
        performance_metrics_table = dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in performance_metrics.columns],
                                data=performance_metrics.to_dict('records'),
                                style_table={"table-layout": "fixed", "width": "auto", 'overflowX': 'hidden'}
        )
        return performance_metrics_table


@app.callback(Output('result', 'data'), 
          [Input('solution_set_dropdown', 'value'),
          Input("input-config","data"),Input('input-mappings', 'data')])
def store_trust_analysis(solution_set_dropdown, config_weights, config_mappings):
    
        if not solution_set_dropdown:
            return None
    
        if not config_weights:
            with open('configs/weights/default.json','r') as f:
                weight_config = json.loads(f.read())
        else:
            weight_config = json.loads(config_weights)
            
        if not config_mappings:
            with open('configs/mappings/default.json', 'r') as f:
                mappings_config = json.loads(f.read())
        else:
            mappings_config = json.loads(config_mappings)
    
            
        test, train, model, factsheet = read_solution(solution_set_dropdown)
    
        final_score, results, properties = get_final_score(model, train, test, weight_config, mappings_config, factsheet, solution_set_dropdown)
        trust_score = get_trust_score(final_score, weight_config["pillars"])
        
        def convert(o):
            if isinstance(o, np.int64): return int(o)  
           
            
        data = {"final_score":final_score,
                "results":results,
                "trust_score":trust_score,
                "properties" : properties}
        return json.dumps(data,default=convert)

for m in fairness_metrics + methodology_metrics + explainability_metrics + robustness_metrics:
    @app.callback(
        [Output("{0}_details".format(m), "is_open"),
        Output("{0}_details".format(m), "style")],
        [Input("toggle_{0}_details".format(m), "n_clicks")],
        [State("{0}_details".format(m), "is_open")],
        prevent_initial_call=True
    )
    def toggle_detail_section(n, is_open):
        if is_open:
            return (not is_open, {'display': 'None'})
        else:
            return (not is_open, {'display': 'Block'})
    
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
       Output('methodology_star_rating', 'children')],
      [Input('result', 'data'),Input("hidden-trigger", "value")])  
def update_figure(data, trig):
      if data is None:
          return [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "", "", "", "", ""]
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
      #bar_chart.update_xaxes(range=[0, 5])
      bar_chart.update_layout(title_text='', title_x=0.5,           paper_bgcolor='#FFFFFF', plot_bgcolor=SECONDARY_COLOR)
      chart_list.append(bar_chart)
     
      #spider
      radar_chart = px.line_polar(r=values, theta=pillars, line_close=True, title='')
      radar_chart.update_layout(title_x=0.5)
      radar_chart.update_traces(fill='toself', fillcolor=TRUST_COLOR, marker_color=TRUST_COLOR, marker_line_color=TRUST_COLOR, marker_line_width=0, opacity=0.6)
      chart_list.append(radar_chart)
     
      #barcharts
      for n, (pillar , sub_scores) in enumerate(results.items()):
          title = "<b style='font-size:32px;''>{}/5</b>".format(final_score[pillar])
          categories = list(map(lambda x: x.replace("_",' ').title(), sub_scores.keys()))
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
          bar_chart_pillar.update_yaxes(range=[0, 5], fixedrange=True)
          #bar_chart_pillar.update_yaxes(fixedrange=True)
          #bar_chart_pillar.update_yaxes(range=[0, 8])
          bar_chart_pillar.update_layout(title_text='', title_x=0.5, xaxis_tickangle=XAXIS_TICKANGLE, paper_bgcolor='#FFFFFF', plot_bgcolor=SECONDARY_COLOR)
            #fig.update_layout(barmode='group', xaxis_tickangle=-45)
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
          radar_chart_pillar = px.line_polar(r=val, theta=categories, line_close=True, title='')
          radar_chart_pillar.update_traces(fill='toself', fillcolor=colors[n], marker_color=colors[n],marker_line_width=1.5, opacity=0.6)
          radar_chart_pillar.update_layout(title_x=0.5)
          chart_list.append(radar_chart_pillar)
      star_ratings = []
      star_ratings.append(show_star_rating(trust_score))
      star_ratings.append(show_star_rating(final_score["fairness"]))
      star_ratings.append(show_star_rating(final_score["explainability"]))
      star_ratings.append(show_star_rating(final_score["robustness"]))
      star_ratings.append(show_star_rating(final_score["methodology"]))
      return chart_list + star_ratings

@app.callback(
    Output("robustness_details", 'children'),
    Input('result', 'data'), prevent_initial_call=True)
def robustness_details(data):
    if not data:
        return []
    result = json.loads(data)
    properties = result["properties"]
    metrics = list(properties["robustness"].keys())

    sections = [html.H3("▶ Robustness Metrics")]
    for i in range(len(metrics)):
        metric_id = metrics[i]
        if properties["robustness"][metric_id] != {}:
            sections.append(create_metric_details_section(metric_id, i, 3))
    return sections

def metric_detail_div(properties):
    prop = []
    for k, v in properties.items():
        prop.append(html.Div("{}: {}".format(v[0], v[1])))
    return html.Div(prop)

@app.callback(
[Output("empirical_robustness_deepfool_attack_details", 'children'), Output("empirical_robustness_deepfool_attack_score", 'children')],
Input('result', 'data'), prevent_initial_call=False)
def Deepfool_Attack_metric_detail(data):
  if data is None:
      return [], []
  else:
      result = json.loads(data)
      properties = result["properties"]
      metric_properties = properties["robustness"]["empirical_robustness_deepfool_attack"]
      metric_scores = result["results"]
      return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["robustness"]["empirical_robustness_deepfool_attack"]))

@app.callback(
[Output("empirical_robustness_carlini_wagner_attack_details", 'children'), Output("empirical_robustness_carlini_wagner_attack_score", 'children')],
Input('result', 'data'), prevent_initial_call=False)
def carlini_wagner_attack_analysis(data):
  if data is None:
      return [], []
  else:
      result = json.loads(data)
      properties = result["properties"]
      metric_properties = properties["robustness"]["empirical_robustness_carlini_wagner_attack"]
      metric_scores = result["results"]
      return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["robustness"]["empirical_robustness_carlini_wagner_attack"]))

@app.callback(
[Output("empirical_robustness_fast_gradient_attack_details", 'children'), Output("empirical_robustness_fast_gradient_attack_score", 'children')],
Input('result', 'data'), prevent_initial_call=False)
def fast_gradient_attack_analysis(data):
  if data is None:
      return [], []
  else:
      result = json.loads(data)
      properties = result["properties"]
      metric_properties = properties["robustness"]["empirical_robustness_fast_gradient_attack"]
      metric_scores = result["results"]
      return metric_detail_div(metric_properties), html.H4("({}/5)".format(metric_scores["robustness"]["empirical_robustness_fast_gradient_attack"]))


@app.callback(
    [Output("confidence_score_details", 'children'), Output("confidence_score_score", 'children')],
    Input('result', 'data'), prevent_initial_call=False)
def confidence_analysis(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["confidence_score"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["robustness"]["confidence_score"]))

@app.callback(
    [Output("loss_sensitivity_details", 'children'), Output("loss_sensitivity_score", 'children')],
    Input('result', 'data'), prevent_initial_call=False)
def loss_sensitivity_analysis(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["loss_sensitivity"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["robustness"]["loss_sensitivity"]))


@app.callback(
    [Output("clever_score_details", 'children'), Output("clever_score_score", 'children')],
    Input('result', 'data'), prevent_initial_call=False)
def clever_score(data):
    if data is None:
        return [], []
    else:
        result = json.loads(data)
        properties = result["properties"]
        metric_properties = properties["robustness"]["clever_score"]
        metric_scores = result["results"]
        return metric_detail_div(metric_properties), html.H4(
            "({}/5)".format(metric_scores["robustness"]["clever_score"]))
 
config_panel.get_callbacks(app)
    
# === LAYOUT ===
layout = html.Div([
    config_panel.layout,
    dbc.Container([
     
        dbc.Row([
            dcc.Store(id='result'),
            
            dbc.Col([html.H1("Analyze", className="text-center")], width=12, className="mb-2 mt-1"),
            
            dbc.Col([dcc.Dropdown(
                    id='solution_set_dropdown',
                    options= get_solution_options(),
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
                    general_section(),
                    trust_section(),
                    pillar_section("fairness", fairness_metrics),
                    pillar_section("explainability", explainability_metrics),
                    pillar_section("robustness", robustness_metrics),
                    pillar_section("methodology", methodology_metrics),
                    dcc.Store(id='training_data'),
                    dcc.Store(id='test_data')
                ], id="analysis_section")
            ],
                width=12, 
                className="mt-2 pt-2 pb-2 mb-2",
                style={
                    "border": "1px solid #d8d8d8",
                    "borderRadius": "6px",
                    "backgroundColor": SECONDARY_COLOR
                }   
            ),
        ], no_gutters=False)
    ])
])
