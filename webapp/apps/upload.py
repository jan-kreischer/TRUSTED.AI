import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from app import app
import os


problem_sets = [{'label': f.name, 'value': f.path} for f in os.scandir('./problem_sets') if f.is_dir()]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def create_info_modal(module_id, name, content):
    modal = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-info-circle"),
            id="{}_info_button".format(module_id), 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(name),
                dbc.ModalBody(content),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close", className="ml-auto", n_clicks=0
                    )
                ),
            ],
            id="{}_info_modal".format(module_id),
            is_open=False,
        ),
    ]
)
    return modal

layout = dbc.Container([
    dbc.Col([

        html.Div([
            create_info_modal("problem_set", "Problem Set", "All different solutions found should belong to the same problem set. It can be seen as the scenario you are working on."),
            html.Div(id="problem_set_alert"),
            html.H3("1. Problem Set"),
            html.H5("Please select the problem set your data belongs to.")
        ], className="text-center"),
        dcc.Dropdown(
            id='problem_set',
            options=problem_sets,
        ),
        html.Div(id='problem_set_path')
    ], 
    className="mb-4"
    ),
    
    dbc.Col([
        html.Div([
            create_info_modal("solution_set", "Solution Set", "One specifically trained model including its training-, test data and factsheet can be seen as a solution set. Your solution set will be saved under the name you entered here."),
            html.Div(id="model_name_alert"),
            html.H3("2. Solution Set"),
            html.H5("Please enter a name for your solution set")
        ], 
        className="text-center"
        ),

        dcc.Input(id="model_name", type="text", placeholder="", value="", debounce=True, style={'width': '100%', 'textAlign': 'center'})
    ], 
    className="mb-4"
    ),
    
    dbc.Col([
        html.Div([
            create_info_modal("training_data", "Training Data", "Please upload the training data you used to train your model. Csv and pickle (pkl) files are accepted. Please place the label to the last column of the dataframe."),
            html.Div(id="training_data_alert"),
            html.H3("3. Training Data"),
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
            create_info_modal("test_data", "Test Data", "Please upload the test data you used to test your model. Csv and pickle (pkl) files are accepted. Please place the label to the last column of the dataframe."),
            html.Div(id="test_data_alert"),
            html.H3("4. Test Data"),
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
    
    # --- FACTSHEET --- #
    
    dbc.Col([
        html.Div([
            create_info_modal("factsheet", "Factsheet", "The factsheet contains the most important information about the methology used."),
            html.Div(id="factsheet_alert"),
            html.H3("5. Factsheet"),
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
            create_info_modal("model", "Model", "Please upload the model you want to assess."),
            html.Div(id="model_alert"),
            html.H3("6. Model"),
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
    html.Div(id="model-uploaded-div", className="text-center"),
    html.Div(html.Span(id="upload_alert")),
    html.Div(dbc.Button("Analyze",  id='trustscore-button', color="primary", className="mt-3"), className="text-center"),
    
],
fluid=False
)



if __name__ == '__main__':
    app.run_server(debug=True)
