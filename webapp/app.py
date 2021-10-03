import dash
import dash_bootstrap_components as dbc
from config import BASE_PATH

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
external_stylesheets = [
    dbc.themes.LUX,
    {
        'href': 'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-50oBUHEmvpQ+1lW4y57PTFmhCaXp0ML5d60M1M7uH2+nqUivzIebhndOJK28anvf',
        'crossorigin': 'anonymous'
    },
    FONT_AWESOME
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, requests_pathname_prefix="{}/".format(BASE_PATH))
app._favicon = app.get_asset_url('favicon.ico')
app.title = "TRUSTED.AI" #Adding html title

server = app.server
app.config.suppress_callback_exceptions = True