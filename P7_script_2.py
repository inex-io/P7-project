#################################################################
######### Projet 7 script 2 - Entrainement du modèle Light GBM ##
######### et description de l'importance globale des variables ##
#################################################################

import pandas as pd
import math
import numpy as np
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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import re
import pickle
import shap


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Fetch Clean and aggregated train data set
def read_file():
    df_train = pd.read_parquet('aggregate_database_train.parquet')
    gc.collect()
    
    return df_train
    
# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(train_df, num_folds, stratified = False, debug= False):
    
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4, #to be adjusted to the setup of the computer nb of Core -1
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )
        # Model fitting and AUC metric calculation
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

        # Best predict proba for train set
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        # Description of feature importance from greatest to smallest
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
 
    display_importances(feature_importance_df)
    explainer = shap.TreeExplainer(clf)
    explainer_0 = explainer.expected_value[0]
    # shap explainer and shap values local storage for future use - script 3
    pickle.dump(explainer_0, open("explainer_0.dat", "wb"))
    shap_values = explainer.shap_values(train_df.iloc[:100000,3:])
    pickle.dump(shap_values, open("shap_values_train.dat", "wb"))

    return feature_importance_df, clf, oof_preds


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

# key numbers computation for future use in Dashboard graph - script 4
def metric(valid_df):
    train_df_graph = pd.read_parquet('train_df_graph.parquet')
    train_df_graph.iloc[0, 9] = 1-np.amax(valid_df)
    train_df_graph.iloc[0, 10] = 1- np.quantile(valid_df, 0.75)
    train_df_graph.iloc[0, 11] = 1- np.quantile(valid_df, 0.25)
    return train_df_graph

def main(debug = False):
    num_rows = 10000 if debug else None
    
    with timer("Chargement du DF aggrégé"):
        train_df = read_file()
        
        gc.collect()
        
    with timer("Run LightGBM with kfold"):
        feat_importance, clf_fitted, valid_df = kfold_lightgbm(train_df, num_folds= 2, stratified= False, debug= debug)
        # local storage of feature importance, train data for future use - script 4
        feat_importance.to_parquet('feat_importance.parquet')
        pickle.dump(clf_fitted, open("clf_fitted.dat", "wb"))
        valid_df_graph = metric(valid_df)
        valid_df_graph.to_parquet('train_df_graph.parquet')
        gc.collect()


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Modèle entraîné"):
        main()