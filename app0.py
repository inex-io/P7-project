import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from app1 import app1


server1 = app1.server

app_dash = Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app_dash.server


app_dash.layout = html.Div([
    html.H1('Dashboard client', style={"text-align": "center"}),
    html.Br(),
    html.Div([
        'Seuil d\'approbation :',
        dcc.Slider(
            0.1,
            1,
            step=0.1,
            value=0.6,
            id='threshold-slider',
        ),
        html.Br(),
        'Numéro du dossier :', # 100002 ko, 100003 ok
        dcc.Input(id='loan-id', value='Saisir un n° de contrat', type='text'),
        #dcc.Link(html.Button(id='submit-button-state', n_clicks=0, children='Submit'), href='/predict', refresh= True),
        html.Button(id='submit-button-state', n_clicks=0, children='Submit'), #href='/predict', refresh= True),
        #dcc.Location(id='url_o', refresh= True)
        html.Div(id='url_o'),
        
], 
    style={'width': '30%', 'vertical-align': 'top', 'display': 'inline-block'}
)
])

@app_dash.callback(
    Output('url_o', 'children'),
    Input('submit-button-state', 'n_clicks'),
    State(component_id='loan-id', component_property='value'),
    State('threshold-slider', 'value')

)    
def output_streaming(n_clicks, input_value, thresh):
    if n_clicks ==0:
        href_id=dcc.Location(id='url', href='/', refresh= False)
    else:
        href_id=dcc.Location(id='url', href='/predict/?id='+input_value+'&thresh='+str(thresh), refresh = True)

    return href_id


@server.route("/")
def home():
    return app_dash.layout

@server.route("/predict/")
def app1_call():
    
    return app1

application = DispatcherMiddleware(
    server,
    {"/predict": app1.server},
)

if __name__ == "__main__":
    run_simple("localhost", 8050, application)