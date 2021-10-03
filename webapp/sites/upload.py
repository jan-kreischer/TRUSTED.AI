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
from joblib import dump, load
from app import app
from config import SCENARIOS_FOLDER_PATH, PICKLE_FILE_EXTENSIONS, JOBLIB_FILE_EXTENSIONS, FACTSHEET_NAME
from helpers import * 

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
    options = []
    if content is not None:
        message = html.Div(name)
        summary, columns = parse_contents(content, name)
        for c in columns:
            options.append({"label": str(c), "value": str(c)})

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
@app.callback(Output('scenario_alert', 'children'),
              [Input('upload_button', 'n_clicks'),
               State('upload_scenario_id', 'value'),
               ], prevent_initial_call=True)
def validate_scenario_id(n_clicks, scenario_id):
    print("scenario_id {}".format(scenario_id))
    if scenario_id is not None:
        return None
    else:
        return html.H6("No scenario was selected", style={"color":"Red"})
  
@app.callback(Output('solution_name_alert', 'children'),
              [Input('upload_button', 'n_clicks'),
               Input('upload_scenario_id', 'value'),
               Input('solution_name', 'value'),
               ], prevent_initial_call=True)
def validate_solution_name(n_clicks, scenario_id, solution_name):
    if n_clicks is not None:
        if not solution_name:
            return html.H6("Please enter a name for your model", style={"color":"Red"})
        else:
            # check if a model with this name already exists for this problem set
            solution_id = name_to_id(solution_name)
            solution_path = get_solution_path(scenario_id, solution_id)
            print("solution_path {}".format(solution_path))
            if os.path.isdir(solution_path):
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
    State('upload_scenario_id', 'value'),
    State('solution_name', 'value')], prevent_initial_call=True)
def save_new_solution_name(n_clicks, scenario_id, solution_name):
    if n_clicks is not None:
        if scenario_id and solution_name:
            print("Uploaded solution set path {}".format(os.path.join(scenario_id, solution_name)))
            return os.path.join(scenario_id, solution_name)
    else:
        return None

@app.callback(Output('upload_alert', 'children'),
              [
               Input('upload_button', 'n_clicks'),
               State('upload_scenario_id', 'value'),
               State('solution_name', 'value'),
               State('general_description', 'value'),
               State('training_data_upload', 'contents'),
               State('training_data_upload', 'filename'),
               State('test_data_upload', 'contents'),
               State('test_data_upload', 'filename'),
               State('target_column_dropdown', 'value'),
               State('factsheet_upload', 'contents'),
               State('factsheet_upload', 'filename'),
               State('model_upload', 'contents'),
               State('model_upload', 'filename')
], prevent_initial_call=True)             
def upload_data(
    n_clicks,
    scenario_id,
    solution_name,
    general_description,
    training_data,
    training_data_filename,
    test_data,
    test_data_filename,
    target_column_name,
    factsheet,
    factsheet_filename,
    model,
    model_filename):
    if n_clicks is None:
        return ""
    else:
        if None in (scenario_id, solution_name, training_data, test_data, model):   
            return html.H5("Please provide all necessary data", style={"color":"Red"},  className="text-center")
        else:
            # Create directory within the problem set to contain the data
            solution_id = name_to_id(solution_name)
            solution_path = get_solution_path(scenario_id, solution_id)
            # Check if directory does not exists yet
            if not os.path.isdir(solution_path):
                os.mkdir(solution_path)
                print("The new directory is created!")
                #return html.H4("Successfully created new directory.", style={"color":"Green"},  className="text-center")
                
                # Upload all the data to the new directory.
                # Saving Training Data
                save_training_data(solution_path, training_data_filename, training_data)
                
                # Saving Test Data
                save_test_data(solution_path, test_data_filename, test_data)
                
                # Saving Factsheet
                save_factsheet(solution_path, FACTSHEET_NAME, factsheet, target_column_name, general_description)
  
                # Saving Model
                save_model(solution_path, model_filename, model)

            else: 
                return html.H4("Directory already exists", style={"color":"Red"}, className="text-center")
                      
            return dcc.Location(pathname="/analyze", id="someid_doesnt_matter")
            return html.H5("Upload Successful", className="text-center")

modals = ["upload_scenario_id", "solution_name", "general_description", "training_data", "test_data", "target_column_name" ,"factsheet", "model"]
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
    
    if file_extension in PICKLE_FILE_EXTENSIONS:
        model = pd.read_pickle(io.BytesIO(decoded))
        pickle.dump(model, open(os.path.join(path, "model" + file_extension), 'wb'))
        
    if file_extension == JOBLIB_FILE_EXTENSIONS:
        model = load(io.BytesIO(decoded))
        dump(model, os.path.join(path, "model" + file_extension)) 
            
# === SITE ===
#scenarios = [{'label': f.name, 'value': f.path} for f in os.scandir(SCENARIOS_FOLDER_PATH) if f.is_dir()]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

layout = dbc.Container([
    dbc.Col([html.H1("Upload", className="text-center")], width=12, className="mb-2 mt-1"),
    
    dbc.Col([
    html.Div([
        html.Div([
            create_info_modal("upload_scenario_id", "Scenario", "All different solutions found should belong to the same scenario", ""),
            html.Div(id="scenario_alert"),
            html.H3(["1. Scenario", html.Sup("*")]),
            html.H5("Please select the scenario your solution belongs to.")
        ], className="text-center"),
        dcc.Dropdown(
            id='upload_scenario_id',
            options=get_scenario_options(),
        ),
    ], 
    className="mb-4"
    ),
    
    html.Div([
        html.Div([
            create_info_modal("solution_name", "Solution", "One specifically trained model including its training-, test data and factsheet can be seen as a solution set. Your solution set will be saved under the name you entered here.", ""),
            html.Div(id="solution_name_alert"),
            html.H3(["2. Solution", html.Sup("*")]),
            html.H5("Please enter a name for your solution.")
        ], 
        className="text-center"
        ),

        dcc.Input(id="solution_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%', 'textAlign': 'center', 'backgroundColor': '#FFFFFF'})
    ], 
    className="mb-4"
    ),
    
    html.Div([
    html.Div([
        create_info_modal("general_description", "Description", "Please add a brief description for your solution.", "*e.g Detect multiple objects within an image, with bounding boxes. The model is trained to recognize 80 different classes of objects in the COCO Dataset. The model consists of a deep convolutional net base model for image feature extraction, together with additional convolutional layers specialized for the task of object detection, that was trained on the COCO data set. It is based on SSD MobileNetV1 using the TensorFlow framework.*"),
        html.H3("3. Description", className="text-center"),
        dcc.Textarea(
            id='general_description',
            value='',
            style={'width': '100%', 'height': 100},
        )], className="mb-4"),
    ]),
    
    html.Div([
        html.Div([
            create_info_modal("training_data", "Training Data", "Please upload the training data you used to train your model. Csv and pickle (pkl) files are accepted. Please place the label to the last column of the dataframe.", ""),
            html.Div(id="training_data_alert"),
            html.H3(["4. Training Data", html.Sup("*")]),
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
            'margin': '10px',
            'backgroundColor': '#FFFFFF'
        })],
        className="mb-4"
    ),
    html.Div(id='training_data_summary'),
    
    # --- TEST DATA UPLOAD --- #
    
    html.Div([
        html.Div([
            create_info_modal("test_data", "Test Data", "Please upload the test data you used to test your model. Csv and pickle (pkl) files are accepted. Please place the label to the last column of the dataframe.", ""),
            html.Div(id="test_data_alert"),
            html.H3(["5. Test Data", html.Sup("*")]),
            html.H5("Please upload the test data"),
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
            'backgroundColor': '#FFFFFF'
        }
    ),
    html.Div(id='test_data_summary'),
            #(csv and pickle files are accepted).
            #"Please place the label to the last column of the dataframe."
        ], className="text-center"),
    ],
            className="mb-4"),
    
    # --- TARGET COLUMN --- #
    
    html.Div([
        html.Div([
            create_info_modal("target_column_name", "Target Column", "The target column contains the values that you want to predict with your model.", ""),
            html.Div(id="target_column_alert"),
            html.H3("6. Target Column"),
            html.H5("Please select the target column"),
                dcc.Dropdown(
        id='target_column_dropdown',
        options=[],
        placeholder='Select Target Column'
    ),
        ], className="text-center"),
    ],
            className="mb-4"),
       
    # --- FACTSHEET --- #
    
    html.Div([
        html.Div([
            create_info_modal("factsheet", "Factsheet", "The factsheet contains the most important information about the methology used.", ""),
            html.Div(id="factsheet_alert"),
            html.H3(["7. Factsheet", html.Sup("*")]),
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
            'margin': '10px',
            'backgroundColor': '#FFFFFF'
        }
    ),
    html.Div(id='factsheet_summary'),
    
    # --- MODEL --- #
    
    html.Div([
        
        html.Div([
            create_info_modal("model", "Model", "Please upload the model you want to assess.", ""),
            html.Div(id="model_alert"),
            html.H3(["8. Model", html.Sup("*")]),
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
            'margin': '10px',
            'backgroundColor': '#FFFFFF'
        }
    ),
    html.Div(id='model_summary'),
    html.Div(html.Span(id="upload_alert")),
    html.Div(dbc.Button("Analyze",  id='upload_button', color="primary", className="mt-3 mb-3"), className="text-center"),
            ], style={
    "border": "1px solid #d8d8d8",
    "borderRadius": "6px",
    "backgroundColor": SECONDARY_COLOR
}),
],
fluid=False
)



if __name__ == '__main__':
    app.run_server(debug=True)
