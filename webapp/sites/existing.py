# === IMPORTS ===
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import json
import os
import shutil
from config import *
from helpers import *
from app import server
from app import app
# === FUNCTIONS ===

def scenario_dropdown_options():
    scenario_ids = list_of_scenarios()
    options = []
    for scenario_id in scenario_ids:
        scenario_name = scenario_id_to_name(scenario_id)
        options.append({"label": scenario_name, "value": scenario_id})
    return options

def scenario_id_to_name(scenario_id):
    return scenario_id.replace("_", " ").title()

def scenario_name_to_id(scenario_name):
    return scenario_name.replace(" ", "_").lower()

def load_scenario(scenario_name):
    """Example Google style docstrings.

    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.

    Example:
        Examples can be given using either the ``Example`` or ``Examples``
        sections. Sections support any reStructuredText formatting, including
        literal blocks::

            $ python example_google.py

    Section breaks are created by resuming unindented text. Section breaks
    are also implicitly created anytime a new section starts.

    Attributes:
        module_level_variable1 (int): Module level variables may be documented in
            either the ``Attributes`` section of the module docstring, or in an
            inline docstring immediately following the variable.

            Either form is acceptable, but the two should not be mixed. Choose
            one convention to document module level variables and be consistent
            with it.

    Todo:
        * For module TODOs
        * You have to also use ``sphinx.ext.todo`` extension

    .. _Google Python Style Guide:
       http://google.github.io/styleguide/pyguide.html

    """
    scenario_path = os.path.join(SCENARIOS_FOLDER_PATH, scenario_name)
    scenario_link = load_scenario_link(scenario_path)
    scenario_description = load_scenario_description(scenario_path)
    scenario_solutions = [f.name for f in os.scandir(os.path.join(SCENARIOS_FOLDER_PATH, scenario_name, SOLUTIONS_FOLDER)) if f.is_dir() and not f.name.startswith('.')]
    return scenario_link, scenario_description, scenario_solutions
  

def display_scenario(scenario_id, scenario_name, scenario_link, scenario_description, scenario_solutions):
    sections = [html.H3("▶ {}".format(scenario_name), style={"text-transform": "capitalize"}),
               html.A("Link", href=scenario_link, className="mt-2 mb-4", style={"font-style": "italic"}),
               html.Div(scenario_description, className="mt-2 mb-4", style={"font-style": "italic"})]
    for i in range(len(scenario_solutions)):
        sections.append(html.H5("-" + scenario_solutions[i], style={"text-transform": "capitalize"}))
    sections.append(html.Hr())
    return html.Div(sections, id="{}_scenario".format(scenario_id))
    """Example Google style docstrings.

    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.

    Example:
        Examples can be given using either the ``Example`` or ``Examples``
        sections. Sections support any reStructuredText formatting, including
        literal blocks::

            $ python example_google.py

    Section breaks are created by resuming unindented text. Section breaks
    are also implicitly created anytime a new section starts.

    Attributes:
        module_level_variable1 (int): Module level variables may be documented in
            either the ``Attributes`` section of the module docstring, or in an
            inline docstring immediately following the variable.

            Either form is acceptable, but the two should not be mixed. Choose
            one convention to document module level variables and be consistent
            with it.

    Todo:
        * For module TODOs
        * You have to also use ``sphinx.ext.todo`` extension

    .. _Google Python Style Guide:
       http://google.github.io/styleguide/pyguide.html

    """
    print("display scenario")
    

def display_scenarios():
    """Example Google style docstrings.

    This module demonstrates documentation as specified by the `Google Python
    Style Guide`_. Docstrings may extend over multiple lines. Sections are created
    with a section header and a colon followed by a block of indented text.

    Example:
        Examples can be given using either the ``Example`` or ``Examples``
        sections. Sections support any reStructuredText formatting, including
        literal blocks::

            $ python example_google.py

    Section breaks are created by resuming unindented text. Section breaks
    are also implicitly created anytime a new section starts.

    Attributes:
        module_level_variable1 (int): Module level variables may be documented in
            either the ``Attributes`` section of the module docstring, or in an
            inline docstring immediately following the variable.

            Either form is acceptable, but the two should not be mixed. Choose
            one convention to document module level variables and be consistent
            with it.

    Todo:
        * For module TODOs
        * You have to also use ``sphinx.ext.todo`` extension

    .. _Google Python Style Guide:
       http://google.github.io/styleguide/pyguide.html

    """
    scenario_ids = list_of_scenarios()
    sections = []
    for scenario_id in scenario_ids:
        scenario_name = scenario_id_to_name(scenario_id)
        scenario_link, scenario_description, scenario_solutions = load_scenario(scenario_id)
        sections.append(display_scenario(scenario_id, scenario_name, scenario_link, scenario_description, scenario_solutions))
    return sections  
  
    
    
#=== CALLBACKS ===
@app.callback(
    Output("create_scenario_dialog", "is_open"),
    [Input("open_create_scenario_dialog", "n_clicks"), Input("submit_create_scenario_dialog", "n_clicks")],
    [State("create_scenario_dialog", "is_open")],
)
def toggle_create_scenario_dialog(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("delete_scenario_dialog", "is_open"),
    [Input("open_delete_scenario_dialog", "n_clicks"), Input("submit_delete_scenario_dialog", "n_clicks")],
    [State("delete_scenario_dialog", "is_open")],
)
def toggle_delete_scenario_dialog(n1, n2, is_open):
    app.logger.info("Open delete")
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [Output("scenario_display", "children"),
     Output("scenario_name", "value"), 
     Output("scenario_link", "value"), 
     Output("scenario_description", "value")],
    [Input('submit_create_scenario_dialog', 'n_clicks')],
    [State('scenario_display', 'children'),
     State('scenario_name', 'value'), 
     State('scenario_link', 'value'), 
     State('scenario_description', 'value')], prevent_initial_call=True)
def create_scenario(n_clicks, scenario_display, scenario_name, scenario_link, scenario_description):
    if scenario_name:
        # Create folder to contain all solutions
        res = os.makedirs(os.path.join(SCENARIOS_FOLDER_PATH, scenario_name, "solutions"))
        f = open(os.path.join(SCENARIOS_FOLDER_PATH, scenario_name, SCENARIO_DESCRIPTION_FILE), "w")
        f.write(scenario_description)
        f.close()
        f = open(os.path.join(SCENARIOS_FOLDER_PATH, scenario_name, SCENARIO_LINK_FILE), "w")
        f.write(scenario_link)
        f.close()
        scenario_id = scenario_name_to_id(scenario_name)
        
        scenario_display = scenario_display + [display_scenario(scenario_id, scenario_name, scenario_link, scenario_description, [])]
        
    return scenario_display, "", "", ""


@app.callback(
    Output("scenario_to_delete", "value"),
    [Input("submit_delete_scenario_dialog", "n_clicks")],
    [State("scenario_to_delete", 'value')], prevent_initial_call=True)
def delete_scenario(n_clicks, path):
    app.logger.info("Deleting {}".format(path))
    try:
        shutil.rmtree(path, ignore_errors=False)
    except Exception as e:
        print(e)
        raise
    return ""
    
    
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("submit", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



# === LAYOUT ===
create_scenario_dialog = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-plus-circle"),
            id="open_create_scenario_dialog", 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Create a scenario"),
                dbc.ModalBody(
                dbc.Form([
    dbc.FormGroup([
        dbc.FormText(
            "A scenario acts as a container for multiple different solutions.",
            color="secondary",
        ),
        dbc.Label("Name", html_for="scenario_name"),
        dbc.Input(type="text", id="scenario_name", placeholder="", debounce=True),
        dbc.Label("Link", html_for="scenario_link"),
        dbc.Input(type="url", id="scenario_link", placeholder="", debounce=True),
        dbc.Label("Description", html_for="scenario_name"),
        dcc.Textarea(
            id='scenario_description',
            value='',
            style={'width': '100%', 'height': 100},
        ),
        dbc.Button(
            "Create", id="submit_create_scenario_dialog", className="ml-auto", n_clicks=0, style={"float": "right"}
        )
    ])
])),
            ],
            id="create_scenario_dialog",
            is_open=False,
        ),
    ]
)


delete_scenario_dialog = html.Div(
    [
        dbc.Button(
            html.I(className="fas fa-minus-circle"),
            id="open_delete_scenario_dialog", 
            n_clicks=0,
            style={"float": "right"}
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Delete a scenario"),
                dbc.ModalBody(dbc.Form([
    dbc.FormGroup([
        dcc.Dropdown(
                    id='scenario_to_delete',
                    options=scenario_dropdown_options(),
                    placeholder='Select Scenario'
        ),
        dbc.Button(
            "Delete", id="submit_delete_scenario_dialog", className="ml-auto", n_clicks=0, style={"float": "right"}
        )
    ])
])),
                
            ],
            id="delete_scenario_dialog",
            is_open=False,
        ),
    ]
)


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                delete_scenario_dialog,
                create_scenario_dialog,
                html.H1("Scenarios", className="text-center", style={"text-transform": "uppercase"}),
            ], width=12),
            dbc.Col(
                html.Div(
                    children=[ 
                        html.Div(children=display_scenarios(), id="scenario_display", style={"backgroundColor": SECONDARY_COLOR}),              
                   ]
                ),
                className="mb-5 mt-5",
                width=12, 
                style={
                    "border": "1px solid #d8d8d8",
                    "borderRadius": "6px",
                    "backgroundColor": SECONDARY_COLOR
                }   
            ),
        ])
    ])

])