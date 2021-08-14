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



# must add this line in order for the app to be deployed successfully on Heroku
from app import server
from app import app
# import all pages in the app
from apps import homepage, upload, visualisation, problem_sets, pillar_fairness, pillar_explainability, pillar_robustness, pillar_methodology, compare
from apps.config_panel import input_ids 
#from apps import *

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button(
                "Search", color="primary", className="ml-2", n_clicks=0
            ),
            width="auto",
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

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
                        dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("Fairness", href="/pillars/fairness"),
                            dbc.DropdownMenuItem("Explainability", href="/pillars/explainability"),
                            dbc.DropdownMenuItem("Robustness", href="/pillars/robustness"),
                            dbc.DropdownMenuItem("Methodology", href="/pillars/methodology"),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="Pillars",
                        ),
                        dbc.NavItem(dbc.NavLink("Demo", href="/upload")),
                        dbc.NavItem(dbc.NavLink("Compare", href="/compare")),
                        dbc.NavItem(dbc.NavLink("Examples", href="/problem-sets")),
                    ], className="ml-auto", navbar=True
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        fluid=False
    ),
    color="#000080",
    dark=True,
    className="mb-4",
)

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
        elif 'pkl' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_pickle(io.BytesIO(decoded))
        df = df.describe().reset_index()
        
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return html.Div([
        html.H5("Statistics regarding "+filename, className="text-center", style={"color":"DarkBlue"}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'scroll'},
        ),
        html.Hr(),
    ])


def save_training_data(path, name, content):
    file_name, file_extension = os.path.splitext(name)
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(path, "train" + file_extension), "wb") as fp:
        fp.write(base64.decodebytes(data))
    
def save_test_data(path, name, content):
    file_name, file_extension = os.path.splitext(name)
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(path, "test" + file_extension), "wb") as fp:
        fp.write(base64.decodebytes(data))
    
def save_model(path, name, content):
    file_name, file_extension = os.path.splitext(name)
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_pickle(io.BytesIO(decoded))
    pickle.dump(df, open(os.path.join(path, "model" + file_extension), 'wb'))

def save_factsheet(path, name):
    app.logger.info(name)
    factsheet = { 'regularization': 'used'}
    with open(os.path.join(path, name), "w",  encoding="utf8") as fp:
        json.dump(factsheet, fp, indent=4)
    
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return homepage.layout
    if pathname == '/upload':
        return upload.layout
    elif pathname == '/visualisation':
        return visualisation.layout
    elif pathname == '/demo':
        return upload.layout
    elif pathname == '/compare':
        return compare.layout
    elif pathname == '/test':
        return  visualisation.layout
    elif pathname == '/pillars/fairness':
        return pillar_fairness.layout
    elif pathname == '/pillars/explainability':
        return pillar_fairness.layout
    elif pathname == '/pillars/robustness':
        return pillar_robustness.layout
    elif pathname == '/pillars/methodology':
        return pillar_methodology.layout
    elif pathname == '/problem-sets':
        return problem_sets.layout
    else:
        return homepage.layout
    
@app.callback(
   Output(component_id='panel', component_property='style'),
   [Input(component_id="toggle-hide", component_property='on')])
def show_hide_element(visibility_state):
    if visibility_state == True:
        return {'display': 'block'}
    if visibility_state == False:
        return {'display': 'none'}

@app.callback([Output('training_data_summary', 'children'),
              Output('training_data_upload', 'children')],
              [Input('training_data_upload', 'contents'),
              State('training_data_upload', 'filename')])
def training_data_preview(content, name):
    if content is not None:
        children = [parse_contents(content, name)]
        return [children, html.Div(['Drag and Drop or Select a Different File (Overwrites the Previous #One)'])]
    return [None, html.Div(['Drag and Drop or Select File'])]


@app.callback([Output('test_data_summary', 'children'),
              Output('test_data_upload', 'children')],
              [Input('test_data_upload', 'contents'),
              State('test_data_upload', 'filename')])
def test_data_preview(content, name):
    if content is not None:
        children = [parse_contents(content, name)]
        return [children, html.Div(['Drag and Drop or Select a Different File (Overwrites the Previous One)'])]
    return [None, html.Div(['Drag and Drop or Select File'])]

@app.callback([Output('factsheet_upload', 'children'),
               Output('factsheet_summary', 'children')],
              [Input('factsheet_upload', 'contents'),
              State('factsheet_upload', 'filename')])
def factsheet_preview(content, name):
    if content is not None:
        message = html.Div(name)
        summary = html.Div()
        return [message, summary]
    return [html.Div(['Drag and Drop or Select File']), None]

@app.callback(
    [Output('model-uploaded-div', 'children'),
    Output('upload-model', 'children')],
    [Input('upload-model', 'contents'),
    State('upload-model', 'filename')])
def model_preview(content, name):
    if content is not None:
        save_model(name, content)
        return [html.H4("Model is uploaded.", style={"color":"Green"}),  html.Div(['Drag and Drop or Select a Different File #(Overwrites the Previous One)'])]
    return [None,  html.Div(['Drag and Drop or Select File'])]

@app.callback([Output('spider', 'style'),
              Output('spider_pillars', 'style'),
              Output('bar', 'style'),
              Output('bar_pillars', 'style')],
              [Input('plot_type', 'value')])
def show_the_graphs(value):
    if value == "spider":
        return [{'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}]
    else:
        return [{'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}]
   
# --- Callbacks --- #
@app.callback(Output('upload_alert', 'children'),
              [
               Input('trustscore-button', 'n_clicks'),
               State('problem_set', 'value'),
               State('model_name', 'value'),
               State('training_data_upload', 'contents'),
               State('training_data_upload', 'filename'),
               State('test_data_upload', 'contents'),
               State('test_data_upload', 'filename'),
               State('factsheet_upload', 'contents'),
               State('factsheet_upload', 'filename'),
               State('model_upload', 'contents'),
               State('model_upload', 'filename')
])             
def upload_data(
    n_clicks,
    problem_set,
    model_name,
    training_data,
    training_data_filename,
    test_data,
    test_data_filename,
    factsheet,
    factsheet_filename,
    model,
    model_filename):
    if n_clicks is None:
        return ""
    else:
        app.logger.info("UPLOAD FUNCTION CALLED")
        app.logger.info(model_name)
        if None in (problem_set, model_name, training_data, test_data, model):   
            return html.H5("Please provide all necessary data", style={"color":"Red"},  className="text-center")
        else:
            # Create directory within the problem set to contain the data
            path = problem_set + "/" + model_name
            app.logger.info(path)
            # Check if directory does not exists yet
            if not os.path.isdir(path):
                os.mkdir(path)
                print("The new directory is created!")
                #return html.H4("Successfully created new directory.", style={"color":"Green"},  className="text-center")
                
                # Upload all the data to the new directory.
                # Saving Training Data
                app.logger.info("Uploading training data")
                save_training_data(path, training_data_filename, training_data)
                
                # Saving Test Data
                app.logger.info("Uploading test data")
                save_test_data(path, test_data_filename, test_data)
                
                # Saving Factsheet
                app.logger.info("Uploading factsheet")
                save_factsheet(path, "factsheet.json")
                    
                # Saving Model
                app.logger.info("Uploading model")
                save_model(path, model_filename, model)   
            else: 
                return html.H4("Directory already exists", style={"color":"Red"}, className="text-center")
                      
            return dcc.Location(pathname="/visualisation", id="someid_doesnt_matter")
            return html.H5("Upload Successful", className="text-center")

# === Callbacks === #
# === Modal Callbacks === #

modals = ["problem_set", "solution_set", "training_data", "test_data", "factsheet", "model"]
for m in modals:
    @app.callback(
        Output("{}_info_modal".format(m), "is_open"),
        [Input("{}_info_button".format(m), "n_clicks"), Input("{}_close".format(m), "n_clicks")],
        [State("{}_info_modal".format(m), "is_open")],
    )
    def toggle_input_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

# === Validation Callbacks === #

@app.callback(Output('problem_set_alert', 'children'),
              [Input('trustscore-button', 'n_clicks'),
               Input('problem_set', 'value'),
               ])
def validate_problem_set(n_clicks, problem_set):
    if n_clicks is not None:
        if problem_set is not None:
            return None
        else:
            return html.H6("No problem set was selected", style={"color":"Red"})
  
@app.callback(Output('model_name_alert', 'children'),
              [Input('trustscore-button', 'n_clicks'),
               Input('problem_set', 'value'),
               Input('model_name', 'value'),
               ])
def validate_model_name(n_clicks, problem_set, model_name):
    if n_clicks is not None:
        if not model_name:
            return html.H6("Please enter a name for your model", style={"color":"Red"})
        else:
            # check if a model with this name already exists for this problem set
            model_path = problem_set + "/" + model_name
            if os.path.isdir(model_path):
                return html.H6("A model with this name already exists", style={"color":"Red"})
            else:  
                return None
            
@app.callback(Output('training_data_alert', 'children'),
               [Input('trustscore-button', 'n_clicks'),
                Input('training_data_upload', 'contents')],
                )
def validate_training_data(n_clicks, training_data):
    if n_clicks is not None:
        if training_data is None:
            return html.H6("No training data uploaded", style={"color":"Red"})
        else:
            return None
        
@app.callback(Output('test_data_alert', 'children'),
               [Input('trustscore-button', 'n_clicks'),
                Input('test_data_upload', 'contents')],
                )
def validate_test_data(n_clicks, test_data):
    if n_clicks is not None:
        if test_data is None:
            return html.H6("No test data uploaded", style={"color":"Red"})
        else:
            return None


@app.callback(Output('factsheet_alert', 'children'),
               [Input('trustscore-button', 'n_clicks'),
                Input('factsheet_upload', 'filename'),
                Input('factsheet_upload', 'contents')
               ])
def validate_factsheet(n_clicks, factsheet_name, factsheet_content):
    app.logger.info("validate factsheet called")
    if n_clicks is not None:
        if factsheet_content is None:
            return html.H6("No factsheet provided", style={"color":"Red"})
        else:
            file_name, file_extension = os.path.splitext(factsheet_name)
            app.logger.info(file_extension)
            if file_extension not in ['.json']:
                return html.H6("Please select a .json file", style={"color":"Red"})   
            return None
        
@app.callback(Output('model_alert', 'children'),
               [Input('trustscore-button', 'n_clicks'),
                Input('model_upload', 'contents')],
                )
def validate_model(n_clicks, model):
    if n_clicks is not None:
        if model is None:
            return html.H6("No model uploaded", style={"color":"Red"})
        else:
            return None
     

visualisation.get_callbacks(app)
#save config

# @app.callback([Output('input-config', 'data'), Output("hidden-trigger", "value")], 
#               [Input('save-weights', 'n_clicks'),
#                Input('apply-config', 'n_clicks')],
#                 list(map(lambda inp: State(inp, "value"),list(filter(lambda ids: ids[:2]=="w_", input_ids)))))
# def store_input_config(n1, n2, *args):
    
    
#     ctx = dash.callback_context
    
#     inputs= dict()
    
#     for name, val in zip(list(filter(lambda ids: ids[:2]=="w_", input_ids)), args):
#         inputs[name[2:]] = float(val)
    
#     config_file =  {
#         "fairness": {
#             "parameters": {
#                 "score_Statistical_Parity": {
#                     "parameter_XY": {
#                         "value": "The value of the parameter_XY",
#                         "description": "The description of the paramter and its impact"
#                     }
#                 }
#             },
#             "weights": {
#                 "Statistical_Parity":  inputs["Statistical_Parity"],
#                 "Disparate_Mistreatment":  inputs["Disparate_Mistreatment"],
#                 "Class_Imbalance":  inputs["Class_Imbalance"],
#                 "Biased_Data":  inputs["Biased_Data"],
#                 "Disparate_Treatment":  inputs["Disparate_Treatment"],
#                 "Disparate_Impact":  inputs["Disparate_Impact"]
#             }
#         },
#         "explainability": {
#             "parameters": {
#                 "score_Algorithm_Class": {
#                     "clf_type_score": {
#                         "value": {
#                             "RandomForestClassifier": 3,
#                             "KNeighborsClassifier": 3,
#                             "SVC": 2,
#                             "GaussianProcessClassifier": 3,
#                             "DecisionTreeClassifier": 4,
#                             "MLPClassifier": 1,
#                             "AdaBoostClassifier": 3,
#                             "GaussianNB": 3.5,
#                             "QuadraticDiscriminantAnalysis": 3,
#                             "LogisticRegression": 3,
#                             "LinearRegression": 3.5
#                         },
#                         "description": "Mapping of Learning techniques to the level of explainability based on on literature research and qualitative analysis of each learning technique. For more information see gh-pages/explainability/taxonomy"
#                     }
#                 },
#                 "score_Feature_Relevance": {
#                     "scale_factor": {
#                         "value": 1.5,
#                         "description": "Used for the calculation to detect outliers in a dataset with the help of quartiels and the Interquartile Range (Q3-Q1) for example the lower bound for outliers is then calculated as follows: lw = Q1-scale_factor*IQR"
#                     },
#                     "distri_threshold": {
#                         "value": 0.6,
#                         "description": "Used for the calulation of how many features make up the a certain fraction (distri_threshold) of all importance. For example if the distri_threshold is 0.6 and the result would be 10% than this would mean that 10% of the used features concentrate 60% of all feature importance, which would mean that the importance of the features is not well balanced where only a few features are important for the classification and the majority of features has only very little or no impact at all"
#                     }
#                 }
#             },
#             "weights": {
#                 "Algorithm_Class": inputs["Algorithm_Class"],
#                 "Correlated_Features":  inputs["Correlated_Features"],
#                 "Model_Size":  inputs["Model_Size"],
#                 "Feature_Relevance": inputs["Feature_Relevance"]
#             }
#         },
#         "robustness": {
#             "parameters": {
#                 "score_Confidence_Score": {
#                     "parameter_XY": {
#                         "value": "The value of the parameter_XY",
#                         "description": "The description of the paramter and its impact"
#                     }
#                 }
#             },
#             "weights": {
#                 "Confidence_Score":  inputs["Confidence_Score"],
#                 "Clique_Method": inputs["Clique_Method"],
#                 "Loss_Sensitivity":  inputs["Loss_Sensitivity"],
#                 "CLEVER_Score":  inputs["CLEVER_Score"],
#                 "Empirical_Robustness_Fast_Gradient_Attack":  inputs["Empirical_Robustness_Fast_Gradient_Attack"],
#                 "Empirical_Robustness_Carlini_Wagner_Attack": inputs["Empirical_Robustness_Carlini_Wagner_Attack"],
#                 "Empirical_Robustness_Deepfool_Attack":  inputs["Empirical_Robustness_Deepfool_Attack"]
#             }
#         },
#         "methodology": {
#             "parameters": {
#                 "score_Normalization": {
#                     "parameter_XY": {
#                         "value": "The value of the parameter_XY",
#                         "description": "The description of the paramter and its impact"
#                     }
#                 }
#             },
#             "weights": {
#                 "Normalization":  inputs["Normalization"],
#                 "Treatment_of_Corrupt_Values":  inputs["Treatment_of_Corrupt_Values"],
#                 "Train_Test_Split":  inputs["Train_Test_Split"],
#                 "Regularization":  inputs["Regularization"],
#                 "Treatment_of_Categorical_Features":  inputs["Treatment_of_Categorical_Features"],
#                 "Feature_Filtering": inputs["Feature_Filtering"]
#             }
#         },
#         "pillars": {
#             "fairness": inputs["fair_pillar"],
#             "explainability": inputs["exp_pillar"],
#             "robustness": inputs["rob_pillar"],
#             "methodology": inputs["meth_pillar"]
#         }
#     }
    
#     return json.dumps(config_file), ctx.triggered[0]['prop_id']
 

# @app.callback(
#     Output("modal-success", "is_open"),
#     [Input("hidden-trigger", "value"),Input("save-success", "n_clicks")],
#     State("modal-success", "is_open"))
# def update_output(trig,n, is_open):
#     if trig == "save-weights.n_clicks" or n:
#         return not is_open
#     else:
#         return is_open
    
# @app.callback(
#     Output("modal-saved", "is_open"),
#     [Input("save-success", "n_clicks")],
#     [State("modal-saved", "is_open"),
#      State("input-config", "data"),
#       State("config-name", "value")])
# def save_config(n_clicks, is_open,config, conf_name):
    
#     if n_clicks and config is not None:
#         config_file = json.loads(config)
#         with open('configs/'+ conf_name+'.json', 'w') as outfile:
#             json.dump(config_file, outfile, indent=4)
#         return not is_open
#     else:
#         return is_open
    

# @app.callback(
#     Output('bar', 'figure'),
#     Input('apply-config', 'n_clicks'))
# def update_figure(n, ):
    
#     final_score, results, properties = get_final_score(model, train_data, test_data, main_config)
#     trust_score = get_trust_score(final_score, main_config["pillars"])
#     performance =  get_performance_table(model, test_data).transpose()
#     pillars = list(final_score.keys())
#     values = list(final_score.values()) 
    
#     bar_chart = go.Figure(data=[go.Bar(
#         x=pillars,
#         y=values,
#         marker_color=my_palette
#             )])

#     return bar_chart
    
if __name__ == '__main__':
    app.run_server(debug=True)