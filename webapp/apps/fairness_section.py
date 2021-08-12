import dash_html_components as html
import dash_bootstrap_components as dbc

fairness_section = html.Div([
    html.H3("Fairness"),
    html.Hr(),
    html.Div(id="class_balance")
    #html.Div(id='path_a'),
    #html.Div(id='analysis_a'),
    #html.Div(id='label_column_name_a'),
    #html.Div(id='protected_column_name_a')
],
    id="fairness_section"
)

