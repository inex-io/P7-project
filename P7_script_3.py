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

def read_test_input(loan_id):
    #df_train = pd.read_parquet('aggregate_database_train.parquet')
    #loan_id = '100005'
    df_test = pd.read_parquet('aggregate_database_test.parquet')
    feats = [f for f in df_test.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]
    X_df_test = df_test[feats][df_test['SK_ID_CURR']== pd.to_numeric(loan_id)]
    gc.collect()
    #return df_train
    return X_df_test

def load_model():
    clf_fitted = pickle.load(open("clf_fitted.dat", "rb"))
    #feature_importance_df = pd.read_parquet('feat_importance.parquet')
    feature_importance_df = pd.read_pickle('feat_importance.pickle')

    return clf_fitted, feature_importance_df

def force_plot (X_test):
    explainer_0 = pickle.load(open("explainer_0.dat", "rb"))
    shap_values = pickle.load(open("shap_values_train.dat", "rb"))
    force_plot_graph_1 = shap.force_plot(explainer_0, shap_values[0][0,0:10], X_test.iloc[0, 0:10], matplotlib= False)
    
    return force_plot_graph_1

def summary_plot(X_test):
    #shap_values = pickle.load(open("shap_values_train.dat", "rb"))
    #summary_plot_graph_1 = shap.summary_plot(shap_values, X_test, max_display=7)
    
    return summary_plot_graph_1

def main(loan_id, debug = False):
    
    #with timer("Chargement du DF de test aggrégé"):
    X_test_df = read_test_input(loan_id)
    X_test = X_test_df.iloc[:, 1:]
    gc.collect()
        
    #with timer("Chargement du modèle et des métriques"):
    clf, feature_importance_df = load_model()
        
        
    #with timer("Run LightGBM with kfold"):
    y_predict = clf.predict_proba(X_test)
    loan_df = X_test_df
    loan_df['TARGET'] = y_predict[0][0]
    force_plot_graph_1 = force_plot (X_test)
    pickle.dump(force_plot_graph_1, open('force_graph_1', 'wb'))
    #summary_plot_graph_1 = summary_plot (X_test)
    #pickle.dump(summary_plot_graph_1, open('summary_plot_graph_1', 'wb'))

    return loan_df
        
    gc.collect()


if __name__ == "__main__":
    #with timer("Modèle entraîné"):
    #loan_id = '100005'
    loan_id = sys.stdin.buffer.read()
    loan_df = main(loan_id)
    #loan_df.to_pickle('loan_results.pickle')
    sys.stdout.buffer.write(loan_df.to_parquet())
    #loan_df.to_parquet('loan_results.parquet')