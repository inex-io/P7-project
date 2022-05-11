import json
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from shap.plots._force_matplotlib import draw_additive_plot
from flask import Flask, request, jsonify, Response
import dash
from dash import Dash, dcc, html, Input, Output, State
import flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
import dash_bootstrap_components as dbc
import pyarrow.parquet as parquet
import pickle
import pandas as pd
import numpy as np
import urllib.parse




app1 = Dash(external_stylesheets = [dbc.themes.BOOTSTRAP], requests_pathname_prefix="/predict/")


feature_importance_df = pd.read_pickle('feat_importance.pickle')
feature_importance = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:10].index


app1.layout = html.Div([
    html.H1('Dashboard client', style={"text-align": "center"}),
    html.Div([
        'Seuil d\'approbation :',
        dcc.Slider(
            0,
            1,
            step=0.1,
            id='threshold-slider',
        ),
        html.Br(),
        #dcc.Input(id='loan-id1'),
        #dcc.Input(id='threshold-slider1'),
        'Numéro du dossier :', # 100002 ko, 100003 ok
        dcc.Input(id='loan-id', type='text'),
        html.Button(id='submit-button-state', n_clicks=1, children='Submit'),
        html.Br(),
        html.Div(id='return-input'),
        html.Br(),
        html.Br(),
        html.Label('Statut :  '),
        html.Span(
                id='target'
        ),
        html.Br(),
        html.Label('Madame/Monsieur :'),
        dcc.RadioItems([
            {'label': 'Madame', 'value': 0, 'disabled': True},
            {'label': 'Monsieur', 'value': 1, 'disabled' : True}
                        ],
            id='gender-code'
        ),

        html.Br(),
        html.Table([
        html.Tr([html.Td('Montant demandé (EUR) :'), html.Td(id='amt-credit')]),
        html.Tr([html.Td('Montant demandé par an (EUR) :'), html.Td(id='amt-annuity')]),
        html.Tr([html.Td('Durée : '), html.Td(id='payment-rate')]),
        html.Tr([html.Td('Age : '), html.Td(id='days-birth')]),
        html.Tr([html.Td('Ancienneté professionnelle : '), html.Td(id='days-employed')]),

        ])       
    ],
        style={'width': '30%', 'vertical-align': 'top', 'display': 'inline-block'}

    ),
    html.Div([
        html.Label('Informations client en relation avec la décision :'),
        html.Br(),
        html.Div(id='force-plot'),
        html.Div([
                html.Br(),
                html.Label('Client vs dossiers approuvé :'),
                dcc.Dropdown(
                    feature_importance,
                    'DAYS_BIRTH',
                    id='summary-x'
                ),
                dcc.Dropdown(
                    feature_importance,
                    'PAYMENT_RATE',
                    id='summary-y'
                ),

                dcc.Graph(
                    id='summary-plot'
                    )
                ]),
    ],
        style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw'},
        className = 'row'
        
    ),
    html.Div([
        html.Div([
            html.Label('Score du client :'),
            dcc.Graph(
                id='score-graph'
                ),
        ], 
            style={'width': '30%', 'horizontal-align': 'center'},
            className = 'row')
    ]),
    html.Div([
        html.Div([
            html.Label('Source #1 :'),
            dcc.Graph(
                id='EXT-SOURCE-1'
                ),
        ], className = 'col-sm'),
        html.Div([
            html.Label('Source #2 :'),
            dcc.Graph(
                id='EXT-SOURCE-2'
                ),
            ], className = 'col-sm'),
        html.Div([        
        html.Label('Source #3 :'),
        dcc.Graph(
            id='EXT-SOURCE-3'
            )
        ], className = 'col-sm')
    ],
        #style={'display': 'inline-block'},
        className = 'row'
    ),
    dcc.Location(id='url_in')
])

@app1.callback(
    Output(component_id='loan-id', component_property='value'),
    Output('threshold-slider', 'value'),
    Output('submit-button-state', 'n_clicks'),    
    Input('url_in', 'href')

)
def init_val(url_in):
    parsed_url = urllib.parse.urlparse(url_in)
    parsed_query = urllib.parse.parse_qs(parsed_url.query)
    input_thresh = str(parsed_query['thresh'][0])
    input_value = int(parsed_query['id'][0])
    
    return input_value, input_thresh, 1

@app1.callback(
    Output(component_id='return-input', component_property='children'),
    Output(component_id='target', component_property='style'),
    Output(component_id='target', component_property='children'),
    Output(component_id='gender-code', component_property='value'),
    Output(component_id='amt-credit', component_property='children'),
    Output(component_id='amt-annuity', component_property='children'),
    Output(component_id='payment-rate', component_property='children'),
    Output(component_id='days-birth', component_property='children'),
    Output(component_id='days-employed', component_property='children'),
    Output(component_id='force-plot', component_property='children'),
    Output(component_id='summary-plot', component_property='figure'),
    Output(component_id='score-graph', component_property='figure'),
    Output(component_id='EXT-SOURCE-1', component_property='figure'),
    Output(component_id='EXT-SOURCE-2', component_property='figure'),
    Output(component_id='EXT-SOURCE-3', component_property='figure'),
    Input('submit-button-state', 'n_clicks'),
    State(component_id='loan-id', component_property='value'),
    State('threshold-slider', 'value'),
    State('summary-x', 'value'),
    State('summary-y', 'value'),
)
def update_output_div(n_clicks, input_value, input_thresh, summary_x, summary_y):
    #input_value = pd.to_numeric(input_value)
    #input_value = '100005' # for debug purpose
    df_input_0, df_train_graph, df_train = read_file(pd.to_numeric(input_value))
    col_sel = ['SK_ID_CURR', 'TARGET', 'CODE_GENDER', 'AMT_CREDIT', 'AMT_ANNUITY', 'PAYMENT_RATE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df_input = df_input_0[col_sel]
    #df_input.iloc[0,1] = 1
    #threshold = 0.1
    threshold = pd.to_numeric(input_thresh)    
    return_value = 'Le numéro de dossier a été trouvé.'
    target = 'Approuvé' if df_input.iloc[0,1] >= threshold else 'Rejeté'
    
    if 'Rejeté' in target:
        colour = {'color' : 'red'}
    else:
        colour = {'color' : 'green'}

    gender = df_input.iloc[0,2]
    amt_credit = df_input.iloc[0,3]
    amt_annuity = df_input.iloc[0,4]
    payment_rate = month_to_yymm(int(1/df_input.iloc[0,5]*12))
    days_birth = (days_to_yymm(df_input.iloc[0,6]))
    days_employed = (days_to_yymm(df_input.iloc[0,7]))

    force_plot_1 = force_plot_call(df_input, input_value)
    score_graph = gauge_graph(df_input.iloc[0,1], df_train_graph.iloc[0,9], df_train_graph.iloc[0,10], df_train_graph.iloc[0,11])
    ext_source_1 = gauge_graph(df_input.iloc[0,8], df_train_graph.iloc[0,0], df_train_graph.iloc[0,1],df_train_graph.iloc[0,2])
    ext_source_2 = gauge_graph(df_input.iloc[0,9], df_train_graph.iloc[0,3], df_train_graph.iloc[0,4],df_train_graph.iloc[0,5])
    ext_source_3 = gauge_graph(df_input.iloc[0,10], df_train_graph.iloc[0,6], df_train_graph.iloc[0,7], df_train_graph.iloc[0,8])
    dff_train = pd.concat([df_train, df_input], axis=0)
    dff_train['TARGET'] = dff_train['TARGET'].astype('str')
    summary_plot= px.scatter(dff_train, x=summary_x, y=summary_y, color= 'TARGET')

    return return_value, colour, target, gender, amt_credit, amt_annuity, payment_rate, days_birth, days_employed, force_plot_1, summary_plot, score_graph, ext_source_1, ext_source_2, ext_source_3



# Fetch train data set, train data for graphics and get results from script 3 
def read_file(input_value):
    df_train_graph = pd.read_parquet('train_df_graph.parquet')
    df_train = pd.read_parquet('aggregate_database_train.parquet')
        #input_value = 100005 # for debug purpose
        #predict_exec = subprocess.run(['python3', 'P7_script_3.py'], input= input_value.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        #df_short = pd.read_parquet(io.BytesIO(predict_exec.stdout))
    df_short = pd.read_parquet('loan_df.parquet')
    #col_sel = ['SK_ID_CURR', 'TARGET', 'CODE_GENDER', 'AMT_CREDIT', 'AMT_ANNUITY', 'PAYMENT_RATE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    #df_short= df_short[col_sel][df_short['SK_ID_CURR'] == input_value]
    df_short= df_short[df_short['SK_ID_CURR'] == input_value]

    return df_short, df_train_graph, df_train


# Routine to transform days in years and months
def days_to_yymm (days_i):
    #days = 197
    days = abs(days_i)
    year = int(days /365)
    month = int((days %year)/365*12)
    return f'{year} ans et {month} mois'

# Routine to transform months in years and months
def month_to_yymm (month_i):
    #month_x = 197
    month_x = abs(month_i)
    year = int(month_x /12)
    month = int((month_x %year))
    return f'{year} ans et {month} mois'


# Call data for Force plot graph
def force_plot_call(df_test, input_value):
    shap.initjs()
    explainer_0 = pickle.load(open("explainer_0.dat", "rb"))
    shap_values = pickle.load(open("shap_values_train.dat", "rb"))
    feats = [f for f in df_test.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]
    X_df_test = df_test[feats][df_test['SK_ID_CURR']== pd.to_numeric(input_value)]
    X_test = X_df_test.iloc[:, 1:]
   
    force_plot_graph_1 = shap.force_plot(explainer_0, shap_values[0][0, 0:9], X_test.iloc[0, 0:9].index, matplotlib= False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_graph_1.html()}</body>"
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "100%", "height": "200px", "border": 0})

# Function to show Gauge graphics
def gauge_graph(x_input, threshold, IQR1, IQR3):

    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = x_input,
    mode = "gauge+number+delta",
    delta = {'reference': threshold},
    gauge = {'axis': {'range': [None, 1]},
             'steps' : [
                 {'range': [0, IQR1], 'color': "lightgray"},
                 {'range': [IQR1, IQR3], 'color': "gray"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}}))
    return fig


if __name__ == '__main__':
    
    app1.run_server(debug=True)