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
from flask import Flask, request, jsonify
from dash import Dash
import dash_bootstrap_components as dbc



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Fetch test data to get loan candidates details
def read_test_input():
    df_test = pd.read_parquet('aggregate_database_test.parquet')
    feats = [f for f in df_test.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]
    X_df_test = df_test[feats]
    gc.collect()

    return X_df_test

# Fetch fitted model and feature importance
def load_model():
    clf_fitted = pickle.load(open("clf_fitted.dat", "rb"))
    #feature_importance_df = pd.read_parquet('feat_importance.parquet')
    feature_importance_df = pd.read_pickle('feat_importance.pickle')

    return clf_fitted, feature_importance_df


def main(debug = False):
    
    X_test_df = read_test_input()
    X_test = X_test_df.iloc[:, 1:]
    gc.collect()
        
    clf, feature_importance_df = load_model()
    
    y_predict = clf.predict_proba(X_test)
    loan_df = X_test_df
    loan_df['TARGET'] = np.array(pd.DataFrame(y_predict).iloc[:,0])
    
    gc.collect()

    return loan_df
        


if __name__ == "__main__":
    #loan_id = '100005' # for debug purpose

    # Get through buffer from script 3 the loan id
        #loan_id = sys.stdin.buffer.read()
        loan_df = main() 
        loan_df.to_parquet('loan_df.parquet')
    # Pass through buffer to script 4 scoring results for the loan id 
        #sys.stdout.buffer.write(loan_df.to_parquet())

