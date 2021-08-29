# -*- coding: utf-8 -*-
import os
import dash_core_components as dcc
import dash_html_components as html
import json
from config import METRICS_CONFIG_PATH

config_explainability,config_pillars = 0, 0
for config in ["config_pillars", "config_explainability"]:
    with open(os.path.join(METRICS_CONFIG_PATH, config + ".json")) as file:
            exec("%s = json.load(file)" % config)

# create panel
exp_panel_comp = []
input_ids = ["w_exp_pillar"]

#weight panel
comp_weight = [html.H4("Weights",style={'text-align':'center'})]
comp_weight.append(html.H5("Pillar Weight",style={'text-align':'center'}))

comp_weight.append(html.Div(dcc.Input(id="w_exp_pillar",value=config_pillars["explainability"], type='text'), 
                            style=dict(display='flex', justifyContent='center')))
comp_weight.append(html.Br())


comp_weight.append(html.H5("Metrics Weights",style={'text-align':'center'}))
for key, val in config_explainability["weights"].items():
    # comp_weight.append(html.Label(key.replace("_",' '))) 
    # comp_weight.append(html.Br())
    # comp_weight.append(dcc.Input(id="w_"+key,value=val, type='text'))
    # comp_weight.append(html.Br())
    input_id = "w_"+key
    input_ids.append(input_id)
    comp_weight.append(html.Div([
        html.Div(html.Label(key.replace("_",' ')), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
        html.Div(dcc.Input(id=input_id, value=val, type='text'), style={'width': '40%', 'display': 'inline-block',"vertical-align": "top",'margin-left': 10}),
        ]))
# parameter panel
exp_panel_comp.append(html.Div(comp_weight))
# exp_panel_comp.append(html.Br())
# exp_panel_comp.append(html.H4("Parameters",style={'text-align':'center'}))
# for key, val in config_explainability["parameters"].items():
#     param_comp = [html.H4(key.replace("_",' '))]
#     for param, info in val.items():
#          input_id = "p_"+param
#          input_ids.append(input_id)  
         
#          param_comp.append(html.Label(html.Strong(key.replace("_",' '))))
#          param_comp.append(html.Br())
#          param_comp.append(dcc.Input(id=input_id,value=str(info["value"]), type='text'))
#          param_comp.append(html.P(info["description"])) 
#     exp_panel_comp.append(html.Div(param_comp))
    

#exported params
explainability_panel = exp_panel_comp
exp_input_ids = input_ids