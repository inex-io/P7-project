#################################################################
######### Projet 7 script 2 - Entrainement du modèle Light GBM ##
######### et description de l'importance globale des variables ##
#################################################################

import pandas as pd
import math
import numpy as np
from numpy import isnan
from sklearn import linear_model ## import du module reg linéaire
from sklearn import preprocessing ## module pour standardiser le jeu de données
from sklearn.naive_bayes import GaussianNB ## module des modèles GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score, confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV, LeaveOneOut, HalvingRandomSearchCV, RandomizedSearchCV, train_test_split
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
    imputer = SimpleImputer(strategy='mean')
    df_train.replace(np.inf, np.nan, inplace = True)
    df_train = pd.DataFrame(imputer.fit_transform(df_train), columns = df_train.columns)
    
    gc.collect()
    
    return df_train
    
# Benchmark of different models through kfold method
def kfold_benchmark(train_df, num_folds, stratified = False, debug= False):
    
    print("Starting classifier Benchmark. Train shape: {}".format(train_df.shape))
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Createb arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    # Resampling to correct unbalanced classes
    over = RandomOverSampler(sampling_strategy=0.1)
    X = train_df.drop(['TARGET'], axis = 1)
    y= train_df['TARGET']
    X, y = over.fit_resample(X, y)
    under = RandomUnderSampler(sampling_strategy=0.5)
    X, y = under.fit_resample(X, y)
    train_df = X
    train_df['TARGET'] = y
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        
        # Naive Bayesian
        Gauss =  GaussianNB()
        gauss_results = cross_val_score(Gauss, train_x, train_y, cv=folds, scoring='accuracy')
        print('Accuracy Gaussian Bayes: %.3f (%.3f)' % (np.mean(gauss_results), np.std(gauss_results))) 
                
        #Log Regression
        logreg =  linear_model.LogisticRegression()
        logreg_results = cross_val_score(logreg, train_x, train_y, cv=folds, scoring='accuracy')
        print('Accuracy Log Reg: %.3f (%.3f)' % (np.mean(logreg_results), np.std(logreg_results))) 
        
        # LGBM classifier
        clf =  LGBMClassifier()
        clf_results = cross_val_score(clf, train_x, train_y, cv=folds, scoring='accuracy')
        print('Accuracy LGBMC: %.3f (%.3f)' % (np.mean(clf_results), np.std(clf_results)))        

        # Random Forest classifier
        rdmfor_clf = RandomForestClassifier()
        rdmfor_clf_results = cross_val_score(rdmfor_clf, train_x, train_y, cv=folds, scoring='accuracy')
        print('Accuracy Random Forest: %.3f (%.3f)' % (np.mean(rdmfor_clf_results), np.std(rdmfor_clf_results)))
     
    
    return clf, clf_results, rdmfor_clf_results, logreg_results, gauss_results

    ### choose of the Ligthgbm ###


# hyper param check
def hyper_param_rand (model, train_df):
    model = LGBMClassifier()
    
    train_x, valid_x, train_y, valid_y = train_test_split( train_df.drop(['TARGET'], axis = 1), train_df['TARGET'],  test_size=0.33, random_state=42)
    
    #train_x = train_df.drop(['TARGET'], axis = 1).iloc[:5000, :]
    #train_y = train_df[:5000]['TARGET']
    #valid_x = train_df.drop(['TARGET'], axis = 1).iloc[5000:, :]
    #valid_y = train_df[5000:]['TARGET']
    
        #distributions = {'n_estimators', 'learning_rate', 'num_leaves=34', 'colsample_bytree', 'subsample', 'max_depth', 'reg_alpha', 'reg_lambda', 'min_split_gain', 'min_child_weight'}
    parameters = {'n_estimators':[10, 100, 10000], 'learning_rate': [0.02, 0.5, 0.1], 'num_leaves' : [10, 34, 50], 'max_depth': [2, 8, 20]}
    
    clf_cv = RandomizedSearchCV(model, parameters, random_state=42).fit(train_x, train_y)
    clf_best_params = clf_cv.best_params_ #{'num_leaves': 34, 'n_estimators': 10000, 'max_depth': 8, 'learning_rate': 0.1}
    return clf_best_params, train_x, valid_x, train_y, valid_y

def cost_function(train_x, valid_x, train_y, valid_y):
    model_opt = LGBMClassifier(num_leaves= 34, n_estimators= 10000, max_depth= 8, learning_rate= 0.01, verbose=-1)
    model_opt.fit(train_x, train_y)
    accuracy_mean = model_opt.score(valid_x, valid_y) #Accuracy : 0.77
    print('Mean accuracy: %.3f' % (accuracy_mean))
    
    # Calculation of predict probability on validation set
    y_pred_lgbmc_opt = model_opt.predict_proba(valid_x)
    y_pred_lgbmc_opt[:, 0] # keep proba of approved candidates

    # Calculate and review of cost function across different approval threshold
    f_beta_table =pd.DataFrame()
    f_beta_x= pd.DataFrame(np.zeros((10, 11)))

    for i in range (0, 10, 1):
        threshold = i/10
        y_pred_thresh = np.where(y_pred_lgbmc_opt[:, 0]>threshold, 0, 1)
        for j in range (0, 10, 1):
            coef= j/10
            f_beta_x[i][j] = fbeta_score(valid_y, y_pred_thresh, beta=coef)
            f_beta_x.max()
    
    # Display of the results 
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
    sns.lineplot(data= f_beta_x, ax=ax[0]).set(xlim=(0, 10), ylim=(0.5, 0.8))
    sns.lineplot(data= f_beta_x, ax=ax[1]).set(xlim=(5, 10), ylim=(0.625, 0.650))
    #ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax[0].set(ylabel= 'F-Beta score', xlabel= 'Threshold * 10')
    ax[0].set_title('F-Beta score by threshold')
    ax[1].set_title('Zoom on F-Beta score for proba >0.5')
    #ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax[1].set(ylabel= 'F-Beta score', xlabel= 'Threshold * 10')
    
    # Beta = 0.6 chosen for the rest of the project
    return f_beta_x, y_pred_thresh
    
def main(debug = False):
    num_rows = 10000 if debug else None
    
    with timer("Dataframe load"):
        train_df = read_file()
        
        gc.collect()
        
    with timer("Run model benchmark with kfold"):
        clf, clf_results, rdmfor_clf_results, logreg_results, gauss_results = kfold_benchmark(train_df, 10)
        gc.collect()

clf= LGBMClassifier()
    with timer("Run Hyperparameters optim"):
        clf_best_params, train_x, valid_x, train_y, valid_y = hyper_param_rand (clf, train_df)
        print (clf_best_params)

    with timer("Run Beta analysis"):
        cost_function(train_x, valid_x, train_y, valid_y)
        
        gc.collect()


if __name__ == "__main__":
    with timer("Modèle entraîné"):
        main()