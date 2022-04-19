#################################################################
######### Projet 7 script 3 - Calcul de la probabilité     ######
######### d'une prédiction coût et                         ######
######### description de l'importance locale des variables ######
#################################################################


import pandas as pd
import math
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import linear_model ## import du module reg linéaire
from sklearn import preprocessing ## module pour standardiser le jeu de données
from sklearn import neighbors ## module des modèles K-NN
from sklearn import cluster ## module des modèles K-means

import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import re
import pickle
import sys
import pyarrow.parquet as parquet
import shap



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Fetch test data to get loan candidates details
def read_test_input(loan_id):
    loan_id = '100005' # use for debug purpose
    df_test = pd.read_parquet('aggregate_database_test.parquet')
    feats = [f for f in df_test.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]
    X_df_test = df_test[feats][df_test['SK_ID_CURR']== pd.to_numeric(loan_id)]
    gc.collect()

    return X_df_test

# Fetch fitted model and feature importance
def load_model():
    clf_fitted = pickle.load(open("clf_fitted.dat", "rb"))
    #feature_importance_df = pd.read_parquet('feat_importance.parquet')
    feature_importance_df = pd.read_pickle('feat_importance.pickle')

    return clf_fitted, feature_importance_df

# Function to display force plot graph
def force_plot (X_test):
    explainer_0 = pickle.load(open("explainer_0.dat", "rb"))
    shap_values = pickle.load(open("shap_values_train.dat", "rb"))
    force_plot_graph_1 = shap.force_plot(explainer_0, shap_values[0][0,0:10], X_test.iloc[0, 0:10], matplotlib= False)
    
    return force_plot_graph_1

def main(loan_id, debug = False):
    
    X_test_df = read_test_input(loan_id)
    X_test = X_test_df.iloc[:, 1:]
    gc.collect()
        
    clf, feature_importance_df = load_model()
    
    y_predict = clf.predict_proba(X_test)
    loan_df = X_test_df
    loan_df['TARGET'] = y_predict[0][0]
    force_plot_graph_1 = force_plot (X_test)
    pickle.dump(force_plot_graph_1, open('force_graph_1', 'wb'))
   

    return loan_df
        
    gc.collect()


if __name__ == "__main__":
    #loan_id = '100005' # for debug purpose

    # Get through buffer from script 3 the loan id
    loan_id = sys.stdin.buffer.read()
    loan_df = main(loan_id) 
    # Pass through buffer to script 4 scoring results for the loan id 
    sys.stdout.buffer.write(loan_df.to_parquet())
