import pandas as pd
import os 
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from itertools import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import mlflow 
from datetime import datetime
import yaml 
from src.logger import logging
from sklearn.pipeline import Pipeline
from datetime import date 

def scale_training_data(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, scaler

config_path = os.path.join('main_config.yml')
with open(config_path, 'r') as f:
    main_config = yaml.safe_load(f)
with open(os.path.join(main_config['data_modelling_config_path']), 'r') as f:
    config = yaml.safe_load(f)

def metrics_report_to_df(model_name, accuracy, dataframe_name, ytrue, ypred,version):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred)
    classification_report = pd.concat(map(pd.DataFrame, [precision, recall, fscore, support]), axis=1)
    classification_report.columns = ["precision", "recall", "f1-score", "support"]
    classification_report.loc['avg/Total', :] = metrics.precision_recall_fscore_support(ytrue, ypred, average='weighted')
    classification_report.loc['avg/Total', 'support'] = classification_report['support'].sum()
    classification_report['model_name'] = model_name
    classification_report['dataframe_name'] = dataframe_name
    classification_report['accuracy'] = accuracy
    classification_report['datetime_stamp']  = str(date.today())
    classification_report['version']  = version
    return classification_report


def k_cross_score_metrics(scoring_metrics, scores,model_name, dataframe_name,version):
    K_scores = pd.DataFrame()
    K_scores['train_time'] = scores['fit_time']
    K_scores['test_time'] = scores['score_time']
    K_scores['test_time'] = scores['score_time']
    for metric in scoring_metrics:
        K_scores[f"k_cross_score_train_{metric}"] = scores[f'train_{metric}']
    for metric in scoring_metrics:
         K_scores[f"k_cross_score_test_{metric}"] = scores[f'test_{metric}']
    K_scores['model_name'] = model_name
    K_scores['dataframe_name'] = dataframe_name
    K_scores['datetime_stamp']  = str(date.today())
    K_scores['version']  = version
    return K_scores


                                       
def modelling_function_experiment_tracking(x_train, x_test, y_train, y_test, 
                                           dataframe_name, k_folds, 
                                           target_column,version):
    version = version 
    final_export = pd.DataFrame()
    k_cross_score_metrics_dataframe = pd.DataFrame()
    model_dict = {'logistic_regression': LogisticRegression(), 
                  'random_forest': RandomForestClassifier(),
                  'Decision_trees':DecisionTreeClassifier(),
                  'xgboost' : xgb.XGBClassifier()}
    train_data = pd.concat([x_train,x_test], ignore_index = True)
    test_data = pd.concat([y_train,y_test], ignore_index = True)
    logging.info(train_data.shape)
    logging.info(test_data.shape)
    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.transform(x_test)
    for values in config['models']['model_list']:
        remote_server_uri = os.getenv("remote_server_uri")
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(str(datetime.now().date()))
        model_name = values
        with mlflow.start_run(run_name = str(str(model_name) + '_' + str(dataframe_name))):
            model = model_dict[model_name]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = accuracy_score(y_test, predictions)
            classification_report = metrics_report_to_df(model_name, score, dataframe_name, y_test, predictions, version)
            final_export = pd.concat([final_export, classification_report])
            precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions,average = 'weighted')
            mlflow.log_param("datetime_stamp", str(date.today()))
            mlflow.log_param("version", version)
            mlflow.log_metric("accuracy",score)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("f1score",fscore)
            pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                ('model',model)])
            k_folds = k_folds
            stratified_kfold = StratifiedKFold(n_splits = k_folds , shuffle=True, random_state=42)
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
            scores = cross_validate(estimator = pipeline, X = train_data, y = test_data, cv=stratified_kfold, 
                                    scoring=scoring_metrics,return_train_score = True)
            k_cross_score_metrics_dataframe_temp = k_cross_score_metrics(scoring_metrics,scores,model_name, dataframe_name,version)
            k_cross_score_metrics_dataframe = pd.concat([k_cross_score_metrics_dataframe, k_cross_score_metrics_dataframe_temp])
            mlflow.log_metric("mean_fit_time_k_cross_training",scores['fit_time'].mean())
            mlflow.log_metric("mean_fit_time_k_cross_testing",scores['score_time'].mean())
            for metric in scoring_metrics:
                mlflow.log_metric(f"k_cross_mean_score_test_{metric}",scores[f'test_{metric}'].mean())
            for metric in scoring_metrics:
                mlflow.log_metric(f"k_cross_mean_score_train_{metric}",scores[f'train_{metric}'].mean())
            mlflow.sklearn.log_model(model,"model_new",registered_model_name = model_name)
    return final_export,k_cross_score_metrics_dataframe,sc