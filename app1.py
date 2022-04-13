from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import subprocess
import pyarrow.parquet as parquet
import pickle
import io
import shap
#import matplotlib.pyplot as plt



app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1('Dashboard client', style={"text-align": "center"}),
    html.Div([
        'Seuil d\'approbation :',
        dcc.Slider(
            0,
            1,
            step=0.1,
            value=0.6,
            id='threshold-slider'
        ),       
        html.Br(),
        html.Div([
                'Numéro du dossier :', # 100002 ko, 100003 ok
                dcc.Input(id='loan-id', value='Saisir un n° de contrat', type='text'),
                html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
                html.Div(id='return-input')
            ]),
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
        #html.Tr([html.Td('Genre :'), html.Td(id='gender-code')]),
        html.Tr([html.Td('Montant demandé (EUR) :'), html.Td(id='amt-credit')]),
        html.Tr([html.Td('Montant demandé par an (EUR) :'), html.Td(id='amt-annuity')]),
        html.Tr([html.Td('Durée : '), html.Td(id='payment-rate')]),
        html.Tr([html.Td('Age : '), html.Td(id='days-birth')]),
        html.Tr([html.Td('Ancienneté professionnelle : '), html.Td(id='days-employed')]),

        ]),
        html.Label('Informations client en relation avec la décision :'),
        html.Div(
            id='force-plot'
            ),
        html.Br(),
        #html.Div(
         #   id='summary-plot'
          #  ),       

    ]),
    html.Div([
        html.Br(),
        html.Label('Score du client :'),
        dcc.Graph(
            id='score-graph'
            ),
        html.Br(),
        html.Br(),
        html.Label('Source #1 :'),
        dcc.Graph(
            id='EXT-SOURCE-1'
            ),
        
        html.Br(),
        html.Label('Source #2 :'),
        dcc.Graph(
            id='EXT-SOURCE-2'
            ),
        html.Br(),
        html.Label('Source #3 :'),
        dcc.Graph(
            id='EXT-SOURCE-3'
            ),
    ]),
])


@app.callback(
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
    #Output(component_id='summary-plot', component_property='children'),
    Output(component_id='score-graph', component_property='figure'),
    Output(component_id='EXT-SOURCE-1', component_property='figure'),
    Output(component_id='EXT-SOURCE-2', component_property='figure'),
    Output(component_id='EXT-SOURCE-3', component_property='figure'),
    Input('submit-button-state', 'n_clicks'),
    State(component_id='loan-id', component_property='value'),
    State('threshold-slider', 'value'),

)
def update_output_div(n_clicks, input_value, threshold_value):
    #input_value = '100005'
    df_input, df_train_graph = read_file(input_value)
    #df_input.iloc[0,1] = 1
    threshold = threshold_value
    #threshold =0.7
    #if df_input.empty:
        #return_value = 'Le numéro de dossier est incorrect.'
        #return return_value

    #else:
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
    #force_plot_1 = force_plot_call()
    force_plot_1 = 'graphique non affiché'
    #summary_plot_graph_1 = summary_plot_call()
    score_graph = gauge_graph(df_input.iloc[0,1], df_train_graph.iloc[0,9], df_train_graph.iloc[0,10], df_train_graph.iloc[0,11])
    ext_source_1 = gauge_graph(df_input.iloc[0,8], df_train_graph.iloc[0,0], df_train_graph.iloc[0,1],df_train_graph.iloc[0,2])
    ext_source_2 = gauge_graph(df_input.iloc[0,9], df_train_graph.iloc[0,3], df_train_graph.iloc[0,4],df_train_graph.iloc[0,5])
    ext_source_3 = gauge_graph(df_input.iloc[0,10], df_train_graph.iloc[0,6], df_train_graph.iloc[0,7], df_train_graph.iloc[0,8])


        #return [df_input.iloc[:,i].tolist() for i in range (1,df_input.T.index.size)]
    return return_value, colour, target, gender, amt_credit, amt_annuity, payment_rate, days_birth, days_employed, force_plot_1, score_graph, ext_source_1, ext_source_2, ext_source_3

def read_file(input_value):
    df_train_graph = pd.read_parquet('train_df_graph.parquet')
    #input_value = '100005'
    predict_exec = subprocess.run(['python3', 'P7_script_3.py'], input= input_value.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    #df_short= pd.read_pickle('loan_results.pickle')
    df_short = pd.read_parquet(io.BytesIO(predict_exec.stdout))
    #df_short= pd.read_parquet('loan_results.parquet')
    #df_short= pd.read_parquet(predict_exec.stdout)
    col_sel = ['SK_ID_CURR', 'TARGET', 'CODE_GENDER', 'AMT_CREDIT', 'AMT_ANNUITY', 'PAYMENT_RATE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df_short= df_short[col_sel]
    
    return df_short, df_train_graph

def days_to_yymm (days_i):
    #days = 197
    days = abs(days_i)
    year = int(days /365)
    month = int((days %year)/365*12)
    return f'{year} ans et {month} mois'

def month_to_yymm (month_i):
    #month_x = 197
    month_x = abs(month_i)
    year = int(month_x /12)
    month = int((month_x %year))
    return f'{year} ans et {month} mois'

def force_plot_call():
    shap.initjs()
    force_plot_graph_1 = pickle.load(open('force_graph_1', 'rb'))
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot_graph_1.html()}</body>"
    
    return html.Iframe(srcDoc=shap_html,
                       style={"width": "100%", "height": "200px", "border": 0})

def summary_plot_call():
    shap.initjs()
    #summary_plot_graph_1 = pickle.load(open('summary_plot_graph_1', 'rb'))
    shap_html_2 = f"<head>{shap.getjs()}</head><body>{summary_plot_graph_1.html()}</body>"
    
    return html.Iframe(srcDoc=shap_html_2,
                       style={"width": "100%", "height": "200px", "border": 0})



def gauge_graph(x_input, threshold, IQR1, IQR3):
    #x_input = 0.08
    #threshold = 0.4
    #IQR1 = 0.6
    #IQR3 = 0.9
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
    
    app.run_server(debug=True)
    #app.run_server(dev_tools_hot_reload=False)
    
