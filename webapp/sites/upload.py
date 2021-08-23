import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import pandas as pd
import json
import pickle
import os
import io
import base64
from app import app
from config import SCENARIOS_FOLDER_PATH
from helpers import list_of_scenarios, create_info_modal

# === CALLBACKS ===
# --- Preview Callbacks --- #
@app.callback([Output('training_data_upload', 'children'),
               Output('training_data_summary', 'children'), 
               Output('target_column_dropdown', 'options')],
              [Input('training_data_upload', 'contents'),
              State('training_data_upload', 'filename')], prevent_initial_call=True)
def training_data_preview(content, name):
    message = html.Div(['Drag and Drop or Select File'])
    summary = []
    if content is not None:
        message = html.Div(name)
        summary, columns = parse_contents(content, name)
        print(columns)
        options = []
        for c in columns:
            options.append({"label": c, "value": c})
    return [message, summary, options]

@app.callback([Output('test_data_upload', 'children'),
               Output('test_data_summary', 'children')],
              [Input('test_data_upload', 'contents'),
              State('test_data_upload', 'filename')], prevent_initial_call=True)
def test_data_preview(content, name):
    message = html.Div(['Drag and Drop or Select File'])
    summary = []
    if content is not None:
        message = html.Div(name)
        summary, columns = parse_contents(content, name)
    return [message, summary]

@app.callback([Output('factsheet_upload', 'children'),
               Output('factsheet_summary', 'children')],
              [Input('factsheet_upload', 'contents'),
              State('factsheet_upload', 'filename')], prevent_initial_call=True)
def factsheet_preview(content, name):
    if content is not None:
        message = html.Div(name)
        summary = html.Div()
        return [message, summary]
    return [html.Div(['Drag and Drop or Select File']), None]

@app.callback([Output('model_upload', 'children'),
               Output('model_summary', 'children')],
              [Input('model_upload', 'contents'),
              State('model_upload', 'filename')], prevent_initial_call=True)
def model_preview(content, name):
    if content is not None:
        message = html.Div(name)
        summary = html.Div()
        return [message, summary]
    return [html.Div(['Drag and Drop or Select File']), None]

# --- Validation Callbacks --- #
@app.callback(Output('problem_set_alert', 'children'),
              [Input('upload_button', 'n_clicks'),
               Input('problem_set', 'value'),
               ], prevent_initial_call=True)
def validate_problem_set(n_clicks, problem_set):
    if n_clicks is not None:
        if problem_set is not None:
            return None
        else:
            return html.H6("No problem set was selected", style={"color":"Red"})
  
@app.callback(Output('model_name_alert', 'children'),
              [Input('upload_button', 'n_clicks'),
               Input('problem_set', 'value'),
               Input('model_name', 'value'),
               ], prevent_initial_call=True)
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
               [Input('upload_button', 'n_clicks'),
                Input('training_data_upload', 'contents')], prevent_initial_call=True
                )
def validate_training_data(n_clicks, training_data):
    if n_clicks is not None:
        if training_data is None:
            return html.H6("No training data uploaded", style={"color":"Red"})
        else:
            return None
        
@app.callback(Output('test_data_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('test_data_upload', 'contents')], prevent_initial_call=True
                )
def validate_test_data(n_clicks, test_data):
    if n_clicks is not None:
        if test_data is None:
            return html.H6("No test data uploaded", style={"color":"Red"})
        else:
            return None


@app.callback(Output('factsheet_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('factsheet_upload', 'filename'),
                Input('factsheet_upload', 'contents')
               ], prevent_initial_call=True)
def validate_factsheet(n_clicks, factsheet_name, factsheet_content):
    if n_clicks is not None:
        if factsheet_content is None:
            return html.H6("No factsheet provided", style={"color":"Red"})
        else:
            file_name, file_extension = os.path.splitext(factsheet_name)
            if file_extension not in ['.json']:
                return html.H6("Please select a .json file", style={"color":"Red"})   
            return None
        
@app.callback(Output('model_alert', 'children'),
               [Input('upload_button', 'n_clicks'),
                Input('model_upload', 'contents')],
                prevent_initial_call=True)
def validate_model(n_clicks, model):
    if n_clicks is not None:
        if model is None:
            return html.H6("No model uploaded", style={"color":"Red"})
        else:
            return None
        
@app.callback(Output('uploaded_solution_set_path', 'data'),
    [Input('upload_button', 'n_clicks'),
    State('problem_set', 'value'),
    State('model_name', 'value')], prevent_initial_call=True)
def save_new_model_name(n_clicks, problem_set_path, model_name):
    if n_clicks is not None:
        if problem_set_path and model_name:
            print("Uploaded solution set path {}".format(os.path.join(problem_set_path, model_name)))
            return {'path': os.path.join(problem_set_path, model_name)}
    else:
        return {'path': ""}

@app.callback(Output('upload_alert', 'children'),
              [
               Input('upload_button', 'n_clicks'),
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
], prevent_initial_call=True)             
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
        if None in (problem_set, model_name, training_data, test_data, model):   
            return html.H5("Please provide all necessary data", style={"color":"Red"},  className="text-center")
        else:
            # Create directory within the problem set to contain the data
            path = problem_set + "/" + model_name
            # Check if directory does not exists yet
            if not os.path.isdir(path):
                os.mkdir(path)
                print("The new directory is created!")
                #return html.H4("Successfully created new directory.", style={"color":"Green"},  className="text-center")
                
                # Upload all the data to the new directory.
                # Saving Training Data
                save_training_data(path, training_data_filename, training_data)
                
                # Saving Test Data
                save_test_data(path, test_data_filename, test_data)
                
                # Saving Factsheet
                save_factsheet(path, "factsheet.json")
                    
                # Saving Model
                save_model(path, model_filename, model)   
            else: 
                return html.H4("Directory already exists", style={"color":"Red"}, className="text-center")
                      
            return dcc.Location(pathname="/analyze", id="someid_doesnt_matter")
            return html.H5("Upload Successful", className="text-center")

modals = ["problem_set", "solution_set", "training_data", "test_data", "target_column" ,"factsheet", "model"]
for m in modals:
    @app.callback(
        Output("{}_info_modal".format(m), "is_open"),
        [Input("{}_info_button".format(m), "n_clicks"), Input("{}_close".format(m), "n_clicks")],
        [State("{}_info_modal".format(m), "is_open")], prevent_initial_call=True
    )
    def toggle_input_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

# === HELPER FUNCTIONS ===
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
        df = df[:8]
        
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ]), []
    
    table = html.Div([
        html.H5("Preview of "+filename, className="text-center", style={"color":"DarkBlue"}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'scroll'},
        ),
        html.Hr(),
    ])
    columns = df.columns.values
    return table, columns


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
    print("file extension {}".format(file_extension))
    
    #if file_extension == ".joblib":
    #    print("saving joblib model")
    
    #if file_extension == ".sav":
    #
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_pickle(io.BytesIO(decoded))
    pickle.dump(df, open(os.path.join(path, "model" + file_extension), 'wb'))

def save_factsheet(path, name):
    factsheet = { 'regularization': 'used'}
    with open(os.path.join(path, name), "w",  encoding="utf8") as fp:
        json.dump(factsheet, fp, indent=4)
         
# === SITE ===
scenarios_folder_path = './scenarios' 
problem_sets = [{'label': f.name, 'value': f.path} for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir()]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = dbc.Container([
    dbc.Col([
        html.Div([
            create_info_modal("problem_set", "Scenario", "All different solutions found should belong to the same scenario", ""),
            html.Div(id="problem_set_alert"),
            html.H3(["1. Scenario", html.Sup("*")]),
            html.H5("Please select the scenario your solution belongs to.")
        ], className="text-center"),
        dcc.Dropdown(
            id='problem_set',
            options=list_of_scenarios(),
        ),
        html.Div(id='problem_set_path')
    ], 
    className="mb-4"
    ),
    
    dbc.Col([
        html.Div([
            create_info_modal("solution_set", "Solution", "One specifically trained model including its training-, test data and factsheet can be seen as a solution set. Your solution set will be saved under the name you entered here.", ""),
            html.Div(id="model_name_alert"),
            html.H3(["2. Solution", html.Sup("*")]),
            html.H5("Please enter a name for your solution.")
        ], 
        className="text-center"
        ),

        dcc.Input(id="model_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%', 'textAlign': 'center'})
    ], 
    className="mb-4"
    ),
    
    dbc.Col([
        html.Div([
            create_info_modal("training_data", "Training Data", "Please upload the training data you used to train your model. Csv and pickle (pkl) files are accepted. Please place the label to the last column of the dataframe.", ""),
            html.Div(id="training_data_alert"),
            html.H3(["3. Training Data", html.Sup("*")]),
            html.H5("Please upload the training data")
        ], className="text-center"),
    dcc.Upload(
        id='training_data_upload',
        children=html.Div([
            'Drag and Drop or Select File'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        })],
        className="mb-4"
    ),
    html.Div(id='training_data_summary'),
    
    # --- TEST DATA UPLOAD --- #
    
    dbc.Col([
        html.Div([
            create_info_modal("test_data", "Test Data", "Please upload the test data you used to test your model. Csv and pickle (pkl) files are accepted. Please place the label to the last column of the dataframe.", ""),
            html.Div(id="test_data_alert"),
            html.H3(["4. Test Data", html.Sup("*")]),
            html.H5("Please upload the test data")
            #(csv and pickle files are accepted).
            #"Please place the label to the last column of the dataframe."
        ], className="text-center"),
    ],
            className="mb-4"),
    
    dcc.Upload(
        id='test_data_upload',
        children=[
            'Drag and Drop or Select a File'
        ],
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='test_data_summary'),
    
    # --- Y COLUMN NAME --- #
    
    dbc.Col([
        html.Div([
            create_info_modal("target_column", "Target Column", "The target column contains the values that you want to predict with your model.", ""),
            html.Div(id="target_column_alert"),
            html.H3("5. Target Column"),
            html.H5("Please select the target column")
            #(csv and pickle files are accepted).
            #"Please place the label to the last column of the dataframe."
        ], className="text-center"),
    ],
            className="mb-4"),
    
    dcc.Dropdown(
                    id='target_column_dropdown',
                    options=[],
                    placeholder='Select Target Column'
    ),
    
    #dcc.Upload(
    #    id='test_data_upload',
    #    children=[
    #        'Drag and Drop or Select a File'
    #    ],
    #    style={
    #        'width': '100%',
    #        'height': '60px',
    #        'lineHeight': '60px',
    #        'borderWidth': '1px',
    #        'borderStyle': 'dashed',
    #        'borderRadius': '5px',
    #        'textAlign': 'center',
    #        'margin': '10px'
    #    }
    #),

    
    # --- FACTSHEET --- #
    
    dbc.Col([
        html.Div([
            create_info_modal("factsheet", "Factsheet", "The factsheet contains the most important information about the methology used.", ""),
            html.Div(id="factsheet_alert"),
            html.H3(["6. Factsheet", html.Sup("*")]),
            html.H5("Please upload the factsheet")
        ], className="text-center"),
    ],
            className="mb-4"),
    
    dcc.Upload(
        id='factsheet_upload',
        children=html.Div([
            'Drag and Drop or Select File'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='factsheet_summary'),
    
    # --- MODEL --- #
    
    dbc.Col([
        html.Div([
            create_info_modal("model", "Model", "Please upload the model you want to assess.", ""),
            html.Div(id="model_alert"),
            html.H3(["7. Model", html.Sup("*")]),
            html.H5("Please upload the model")
        ], className="text-center")
        # 5. Please upload the model as a .sav file.
    ], className="mb-4"),
    dcc.Upload(
        id='model_upload',
        children=html.Div([
            'Drag and Drop or Select File'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    html.Div(id='model_summary'),
    html.Div(html.Span(id="upload_alert")),
    html.Div(dbc.Button("Analyze",  id='upload_button', color="primary", className="mt-3"), className="text-center"),
    
],
fluid=False
)



if __name__ == '__main__':
    app.run_server(debug=True)
