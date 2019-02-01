#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains functions to help scrape
"""
from datetime import date, timedelta
import json_schedule
import pandas as pd
import numpy as np
from geopy.distance import vincenty
import requests
import scrape_functions
from sklearn.metrics import log_loss
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import sklearn.metrics
from sklearn.calibration import CalibratedClassifierCV
import pickle
import boto3
import io
import os
import seaborn as sbs
import pandas as pd
import time
import numpy as np
import warnings
import elo
import s3fs
from datetime import datetime
from sklearn.metrics import log_loss
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import sklearn.metrics
from sklearn.calibration import CalibratedClassifierCV
import pickle
import boto3
import io


def write_boto_s3(df, bucket, filename):
    """
    write csv file to s3
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, filename).put(Body=csv_buffer.getvalue())
    print("S3 " + str(bucket) + "/" + str(filename) + " updated")


def s3_model_object_load(bucket, location):
    s3 = boto3.resource('s3')
    with io.BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(location, data)
        data.seek(0)  # move back to the beginning after writing
        m = pickle.load(data)
    return (m)

def s3_model_object_dump(model, bucket, filename):
    s3_resource = boto3.resource('s3')
    pickle_byte_obj = pickle.dumps(model)
    s3_resource.Object(bucket, filename).put(Body=pickle_byte_obj)

def implied_pct_to_ml(prob):
    if prob > 0.5:
        ml = (prob / (1 - prob)) * (-100)
    else:
        ml = ((1 - prob) / prob) * 100
    return(ml)


def ml_to_prob(ml):
    if ml > 0:
        implied_prob = 100 / (ml + 100)
    else:
        implied_prob = (-ml) / (-ml + 100)
    return (implied_prob)

## Model Accuracy and Log Loss Function
def model_complete_scoring(model, bucket, filename, test_X, test_Y, pred_type="proba"):
    """
    :param model:
    :param filename:
    :param train_X:
    :param train_Y:
    :param test_X:
    :param test_Y:
    :param pred_type:
    :return:
    """
    # Save Model
    s3_model_object_dump(model, bucket, filename)

    # Obtaining test accuracy
    acc = model.score(test_X, test_Y)
    # Predict on the test set
    if pred_type == "proba":
        pred = pd.DataFrame(model.predict_proba(test_X)).iloc[:, 1]
    else:
        pred = pd.Series(model.predict(test_X))
    # Score test Logloss
    logloss = log_loss(test_Y, pred)
    # Brier score
    brier_score = sklearn.metrics.brier_score_loss(test_Y, pred)

    # Calibration Correlation
    calibration_df = pd.concat([round(pred, 2).reset_index(drop=True), pd.Series(test_Y).reset_index(drop=True)],
                               axis=1)
    calibration_df.columns = ['pred', 'home_win']
    calibration_agg = calibration_df.groupby('pred')['home_win'].agg(['mean', 'count']).reset_index()

    calibration_corr = wcorr(np.array(calibration_agg['pred']),
                             np.array(calibration_agg['mean']), np.array(calibration_agg['count']))

    return (list([acc, logloss, brier_score, calibration_corr, pred]))


def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def wcorr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def read_boto_s3(bucket, file):
    """
    read file from s3 to pandas
    :return:
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=file)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return (df)


def model_game_data(model_df,
                    control_windows=[10],
                    control_offsets=[48],
                    result_windows=[20, 40],
                    result_offsets=[48, 168],
                    control_features=['starter_wa_hours_rest', 'starter_wa_travel_km'],
                    elo_metrics=['elo_k4_wm4', 'elo_k4_wm4_SO2', 'elo_k4_wm2_SO2'],
                    model_list=['lr_cvsearch', 'gnb_isotonic', 'rf_isotonic', 'mlp_isotonic', 'lr_model'],
                    result_features=['wa_game_GF', 'wa_game_GA', #'wa_game_SF', 'wa_game_SA',
                          'wa_game_xGF_adj','wa_game_xGA_adj',
                          'wa_game_SOGF_rate','wa_game_SOGA_rate','wa_game_FwdF_share','wa_game_FwdA_share',
                          'wa_game_PPGF', 'wa_game_PKGA', 'wa_game_PPAtt','wa_game_PKAtt',
                          'wa_game_skater_sim','starter_wa_PP_svPct','starter_wa_svPct'],
                    standard_features=['hours_rest', 'travel_km']):
    all_game_probs = model_df.loc[:, ['home_win', 'status', 'id', 'season', 'shootout_final', 'home_team', 'away_team',
                                      'away_starter_id', 'home_starter_id', 'win_margin']]

    ######
    # Splitting data into X and Y
    ######
    complete_feature_set = [str(feature) + "_w" + str(window) + "_o" + str(offset)
                            for feature in result_features
                            for window in [10, 20, 40]
                            for offset in [48, 168]] + \
                           [str(feature) + "_w" + str(window) + "_o" + str(offset)
                            for feature in control_features
                            for window in [10]
                            for offset in [24, 48]] + elo_metrics + standard_features

    complete_feature_list = [str(venue) + "_" + feat for venue in ['home', 'away'] for feat in complete_feature_set]

    # Subset training data to Final games
    model_df_final = model_df.loc[model_df['status'] == 'Final', :]

    train_X, test_X, train_Y, test_Y, train_Szn, test_Szn = train_test_split(
        model_df_final.loc[:, complete_feature_list],
        model_df_final.loc[:, 'home_win'],
        model_df_final.loc[:, 'season'],
        stratify=model_df.loc[:, 'home_win'],
        test_size=0.3, random_state=42)

    ## Validate over seasons
    gkf = GroupKFold(n_splits=len(train_Szn.unique()))

    ## Scale data to train set
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # Save scaler
    s3_resource = boto3.resource('s3')
    pickle_byte_obj = pickle.dumps(scaler)
    s3_resource.Object('games-all', 'Models/game_features_scaler').put(Body=pickle_byte_obj)

    train_df = pd.DataFrame(train_X, columns=complete_feature_list)
    test_df = pd.DataFrame(test_X, columns=complete_feature_list)

    ## Scale full dataframe
    full_df = model_df.loc[:, complete_feature_list]
    full_df = scaler.transform(full_df)
    full_df = pd.DataFrame(full_df, columns=complete_feature_list)

    ######
    # Model initializations
    ######
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200,
                                max_depth=80,
                                criterion='entropy',
                                class_weight='balanced',
                                min_samples_leaf=3,
                                min_samples_split=8,
                                random_state=24)
    rf_isotonic = CalibratedClassifierCV(rf, cv=5, method='isotonic')

    # MultiLayer Perceptron
    mlp = MLPClassifier(learning_rate='adaptive',
                        hidden_layer_sizes=(10, 10, 10))
    mlp_isotonic = CalibratedClassifierCV(mlp, cv=5, method='isotonic')

    ## Gaussian Naive-Bayes (with prior)
    home_win_prob = (model_df['home_win'].mean())
    gnb = GaussianNB(priors=[1 - home_win_prob, home_win_prob])
    gnb_isotonic = CalibratedClassifierCV(gnb, cv=5, method='isotonic')

    # XGBoost
    xgb_init = XGBRegressor(objective='binary:logistic', alpha=10, n_estimators=100)

    xgb_test_params = {'colsample_bytree': [0.3, 0.5, 0.7], 'learning_rate': [0.05, 0.1],
                       'max_depth': [5, 10, 25]}
    xgb_model = sklearn.model_selection.RandomizedSearchCV(estimator=xgb_init,
                                                           param_distributions=xgb_test_params,
                                                           cv=gkf,
                                                           n_iter=10)

    # Logistic Regression
    lr_init = sklearn.linear_model.LogisticRegression()

    lr_param_grid = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    ## LR Search
    lr_cvsearch = sklearn.model_selection.GridSearchCV(lr_init, lr_param_grid, scoring='neg_log_loss', cv=10, verbose=0,
                                                       n_jobs=-1)

    ## LM Search
    lr_model = sklearn.model_selection.RandomizedSearchCV(estimator=lr_init,
                                                          param_distributions=lr_param_grid,
                                                          cv=gkf,
                                                          random_state=1, n_iter=10, verbose=0, n_jobs=-1)

    all_results = pd.DataFrame()
    test_predictions = pd.DataFrame()

    for result_window in result_windows:
        for control_window in control_windows:
            for result_offset in result_offsets:
                for control_offset in control_offsets:
                    for metric in elo_metrics:

                        feature_set = [str(feature) + "_w" + str(result_window) + "_o" + str(result_offset)
                                       for feature in result_features] + \
                                      [str(feature) + "_w" + str(control_window) + "_o" + str(control_offset)
                                       for feature in control_features] + standard_features + [metric]

                        feature_list = [str(venue) + "_" + feat for venue in ['home', 'away'] for feat in feature_set]

                        full_scaled = full_df.loc[:, feature_list]

                        # Model DF
                        train_X0 = train_df.loc[:, feature_list].as_matrix()
                        test_X0 = test_df.loc[:, feature_list].as_matrix()

                        # LR1
                        lr_cvsearch.fit(train_X0, train_Y)

                        lr_cvsearch_out = model_complete_scoring(lr_cvsearch, 'games-all',
                                                                 'Models/lr_cvsearch_' + str(metric) + '_w' + str(
                                                                     result_window) + '_o' + str(
                                                                     result_offset), test_X0, test_Y, "proba")

                        # Gaussian Naive-Bayes
                        gnb_isotonic.fit(train_X0, train_Y)

                        gnb_isotonic_out = model_complete_scoring(gnb_isotonic, 'games-all',
                                                                  'Models/gnb_isotonic_' + str(metric) + '_w' + str(
                                                                      result_window) + '_o' + str(result_offset),
                                                                  test_X0, test_Y, "proba")

                        # MultiLayer Perceptron
                        mlp_isotonic.fit(train_X0, train_Y)

                        mlp_isotonic_out = model_complete_scoring(mlp_isotonic, 'games-all',
                                                                  'Models/mlp_isotonic_' + str(metric) + '_w' + str(
                                                                      result_window) + '_o' + str(result_offset),
                                                                  test_X0, test_Y, "proba")

                        # Random Forest
                        rf_isotonic.fit(train_X0, train_Y)

                        rf_isotonic_out = model_complete_scoring(rf_isotonic, 'games-all',
                                                                 'Models/rf_isotonic_' + str(metric) + '_w' + str(
                                                                     result_window) + '_o' + str(
                                                                     result_offset),
                                                                 test_X0, test_Y, "proba")

                        # XGBoost
                        xgb_model.fit(train_X0, train_Y, groups=train_Szn)

                        xgb_model_out = model_complete_scoring(xgb_model, 'games-all',
                                                               'Models/xgb_model_' + str(metric) + '_w' + str(
                                                                   result_window) + '_o' + str(
                                                                   result_offset),
                                                               test_X0, test_Y, "predict")

                        # Logisitc model
                        lr_model.fit(train_X0, train_Y, groups=train_Szn)

                        lr_model_out = model_complete_scoring(lr_model, 'games-all',
                                                              'Models/lr_model_' + str(metric) + '_w' + str(
                                                                  result_window) + '_o' + str(
                                                                  result_offset),
                                                              test_X0, test_Y, "proba")

                        blend_pred = pd.DataFrame(dict(lr_pred=lr_model_out[4],
                                                       xgb_pred=xgb_model_out[4],
                                                       gnb_pred=gnb_isotonic_out[4],
                                                       mlp_pred=gnb_isotonic_out[4],
                                                       rf_pred=rf_isotonic_out[4]
                                                       )).mean(axis=1)

                        blend_logloss = log_loss(test_Y, blend_pred)
                        blend_brier_score = sklearn.metrics.brier_score_loss(test_Y, blend_pred)

                        # Calibration Correlation
                        calibration_df = pd.concat(
                            [round(blend_pred, 2).reset_index(drop=True), pd.Series(test_Y).reset_index(drop=True)],
                            axis=1)
                        calibration_df.columns = ['pred', 'home_win']
                        calibration_agg = calibration_df.groupby('pred')['home_win'].agg(
                            ['mean', 'count']).reset_index()

                        blend_calibration_corr = wcorr(np.array(calibration_agg['pred']),
                                                       np.array(calibration_agg['mean']),
                                                       np.array(calibration_agg['count']))

                        cols = ["acc", "logloss", "brier_score", "cal_corr"]

                        model_results = pd.concat(
                            [pd.DataFrame(np.array(
                                [metric, result_window, result_offset, blend_logloss, blend_brier_score,
                                 blend_calibration_corr]).reshape(1,
                                                                  6),
                                          columns=['metric', 'window', 'offset', 'blend_logloss', 'blend_brier_score',
                                                   'blend_cal_corr']),
                             pd.DataFrame(np.array(lr_cvsearch_out[:4]).reshape(1, 4),
                                          columns=["lrsearch_" + str(col) for col in cols]),
                             pd.DataFrame(np.array(gnb_isotonic_out[:4]).reshape(1, 4),
                                          columns=["gnb_" + str(col) for col in cols]),
                             pd.DataFrame(np.array(mlp_isotonic_out[:4]).reshape(1, 4),
                                          columns=["mlp_" + str(col) for col in cols]),
                             pd.DataFrame(np.array(rf_isotonic_out[:4]).reshape(1, 4),
                                          columns=["rf_" + str(col) for col in cols]),
                             pd.DataFrame(np.array(lr_model_out[:4]).reshape(1, 4),
                                          columns=["lr_" + str(col) for col in cols]),
                             pd.DataFrame(np.array(xgb_model_out[:4]).reshape(1, 4),
                                          columns=["xgb_" + str(col) for col in cols])
                             ], axis=1)

                        all_results = all_results.append(model_results)

                        write_boto_s3(all_results, 'games-all', 'prediction_results_param.csv')

                        test_predictions = pd.concat(
                            [pd.DataFrame(np.array([metric, result_window, result_offset]).reshape(1, 3)),
                             test_Y.reset_index(drop=True),
                             lr_cvsearch_out[4].reset_index(drop=True),
                             gnb_isotonic_out[4].reset_index(drop=True),
                             mlp_isotonic_out[4].reset_index(drop=True),
                             rf_isotonic_out[4].reset_index(drop=True),
                             lr_model_out[4].reset_index(drop=True),
                             xgb_model_out[4].reset_index(drop=True),
                             blend_pred.reset_index(drop=True)
                             ], axis=1).fillna(method='ffill')

                        test_predictions.columns = ['metric', 'window', 'offset', 'home_win', 'lrsearch_pred',
                                                    'gnb_pred',
                                                    'mlp_pred', 'rf_pred', 'lr_pred', 'xgb_pred', 'blend_pred']

                        test_predictions = test_predictions.append(test_predictions)

                        write_boto_s3(test_predictions, 'games-all', 'test_results_param.csv')

                        ######
                        ## Ensemble models
                        ######
                        for model in model_list:
                            # Load models
                            loaded_model = s3_model_object_load('games-all',
                                                                'Models/' + str(model) + "_" + str(metric) + '_w' + str(
                                                                    result_window) + '_o' + str(result_offset))

                            # Try
                            if model != 'xgb_model':
                                pred = pd.DataFrame(loaded_model.predict_proba(full_scaled)).iloc[:, 1]
                            else:
                                pred = pd.Series(loaded_model.predict(full_scaled))

                            pred = pred.to_frame()
                            pred.columns = [
                                str(model) + "_" + str(metric) + '_w' + str(result_window) + '_o' + str(result_offset)]
                            all_game_probs = pd.concat([all_game_probs.reset_index(drop=True),

                                                        pred], axis=1)

    write_boto_s3(all_game_probs, 'games-all', 'all_game_probs.csv')

    return (all_game_probs)


def score_game_data(model_df,
                    control_windows=[10],
                    control_offsets=[48],
                    result_windows=[20, 40],
                    result_offsets=[48, 168],
                    starter_control_standard=['starter_hours_rest', 'starter_travel_km'],
                    starter_control_weighted=['starter_wa_hours_rest', 'starter_wa_travel_km'],
                    elo_metrics=['elo_k4', 'elo_k4_wm4', 'elo_k4_wm4_SO2', 'elo_k4_wm2_SO2', 'elo_k8_wm8_SO2'],
                    model_list=['lr_cvsearch', 'gnb_isotonic', 'rf_isotonic', 'mlp_isotonic', 'lr_model', 'xgb_model'],
                    custom_model_list=['gnb_isotonic_custom', 'lr_cvsearch_custom', 'mlp_isotonic_custom',
                                       # 'rf_isotonic_custom',
                                       'lr_model_custom', 'svm_model_custom', 'xgb_model_custom'],
                    control_features=['wa_travel_km', 'wa_hours_rest'],
                    result_features=['wa_game_GF', 'wa_game_GA',  # 'wa_game_SF', 'wa_game_SA',
                                     'wa_game_xGF_adj', 'wa_game_xGA_adj',
                                     'wa_game_SOGF_rate', 'wa_game_SOGA_rate', 'wa_game_FwdF_share',
                                     'wa_game_FwdA_share',
                                     'wa_game_PPGF', 'wa_game_PKGA', 'wa_game_PPAtt', 'wa_game_PKAtt',
                                     'wa_game_skater_sim', 'starter_wa_PP_svPct', 'starter_wa_svPct'],
                    standard_features=['hours_rest', 'travel_km']):
    scaler = s3_model_object_load('games-all', 'Models/game_features_scaler')

    complete_feature_set = [str(feature) + "_w" + str(window) + "_o" + str(offset)
                            for feature in result_features
                            for window in [10, 20, 40]
                            for offset in [48, 168]] + \
                           [str(feature) + "_w" + str(window) + "_o" + str(offset)
                            for feature in control_features
                            for window in [10]
                            for offset in [48]] + \
                           [str(feature) + "_w" + str(window) + "_o" + str(offset)
                            for feature in starter_control_weighted
                            for window in [10]
                            for offset in [24, 48]] + \
                           elo_metrics + standard_features + starter_control_standard

    complete_feature_list = [str(venue) + "_" + feat for venue in ['home', 'away'] for feat in complete_feature_set]

    ## Scale full dataframe
    full_df = model_df.loc[:, complete_feature_list]
    print(full_df.shape)

    full_df = scaler.transform(full_df)
    full_df = pd.DataFrame(full_df, columns=complete_feature_list)

    all_game_probs = model_df.loc[:, ['home_win', 'status', 'id', 'season', 'home_team', 'away_team', 'away_starter_id',
                                      'home_starter_id', 'shootout_final', 'win_margin']]

    ######
    ## Load models and score
    ######
    for result_window in result_windows:
        for control_window in control_windows:
            for result_offset in result_offsets:
                for control_offset in control_offsets:
                    for metric in elo_metrics:
                        for model in model_list:
                            # Load models
                            print(
                                str(model) + "_" + str(metric) + '_w' + str(result_window) + '_o' + str(result_offset))
                            loaded_model = s3_model_object_load('games-all',
                                                                'Models/' + str(model) + "_" + str(metric) + '_w' + str(
                                                                    result_window) + '_o' + str(result_offset))

                            feature_set = [str(feature) + "_w" + str(result_window) + "_o" + str(result_offset)
                                           for feature in result_features] + \
                                          [str(feature) + "_w" + str(control_window) + "_o" + str(control_offset)
                                           for feature in control_features] + standard_features + [metric]

                            feature_list = [str(venue) + "_" + feat for venue in ['home', 'away'] for feat in
                                            feature_set]

                            full_scaled = full_df.loc[:, feature_list]

                            # Try
                            if model != 'xgb_model':
                                pred = pd.DataFrame(loaded_model.predict_proba(full_scaled)).iloc[:, 1]
                            else:
                                pred = pd.Series(loaded_model.predict(full_scaled))

                            pred = pred.to_frame()
                            pred.columns = [
                                str(model) + "_" + str(metric) + '_w' + str(result_window) + '_o' + str(result_offset)]
                            all_game_probs = pd.concat([all_game_probs.reset_index(drop=True), pred], axis=1)

    game_features_custom = s3_model_object_load('games-all', 'Models/game_features_custom')

    scaler = s3_model_object_load('games-all', 'Models/game_features_scaler_custom')

    ## Scale full dataframe
    full_custom_df = model_df.loc[:, game_features_custom]
    full_custom_df = scaler.transform(full_custom_df)
    full_custom_df = pd.DataFrame(full_custom_df, columns=game_features_custom)

    ######
    ## Score data
    ######
    for model in custom_model_list:
        print(model)
        # Load models
        loaded_model = s3_model_object_load('games-all', 'Models/' + str(model))

        # Try
        if model != 'xgb_model_custom':
            pred = pd.DataFrame(loaded_model.predict_proba(full_custom_df.as_matrix())).iloc[:, 1]
        else:
            pred = pd.Series(loaded_model.predict(full_custom_df.as_matrix()))

        pred = pred.to_frame()
        pred.columns = [str(model)]
        all_game_probs = pd.concat([all_game_probs.reset_index(drop=True),
                                    pred], axis=1)

    return (all_game_probs)


def ensemble_models(all_game_probs,
                    model_list=['lr_cvsearch', 'gnb_isotonic', 'rf_isotonic', 'mlp_isotonic', 'lr_model'],
                    custom_model_list=['gnb_isotonic_custom', 'lr_cvsearch_custom', 'mlp_isotonic_custom',
                                       # 'rf_isotonic_custom',
                                       'lr_model_custom', 'svm_model_custom', 'xgb_model_custom'],
                    result_windows=[20, 40],
                    result_offsets=[48, 168],
                    elo_metrics=['elo_k4_wm4', 'elo_k4_wm4_SO2', 'elo_k4_wm2_SO2']):
    ensembled_game_probs = all_game_probs.loc[:,
                           ['home_win', 'status', 'id', 'season', 'home_team', 'away_team', 'away_starter_id',
                            'home_starter_id', 'shootout_final', 'win_margin']]

    feature_set = [str(model) + "_" + str(metric) + "_w" + str(window) + "_o" + str(offset)
                   for model in model_list
                   for metric in elo_metrics
                   for window in result_windows
                   for offset in result_offsets] + custom_model_list

    feature_set = [feat for feat in feature_set if feat in all_game_probs.columns]

    ensemble_list = ['lr_model_randomsearch', 'lr_model_cvsearch', 'lr_model_regularized']

    all_game_probs_final = all_game_probs.loc[all_game_probs['status'] == 'Final', :]

    train_X, test_X, train_Y, test_Y, train_Szn, test_Szn = train_test_split(all_game_probs_final.loc[:, feature_set],
                                                                             all_game_probs_final.loc[:, 'home_win'],
                                                                             all_game_probs_final.loc[:, 'season'],
                                                                             stratify=all_game_probs_final.loc[:,
                                                                                      'home_win'],
                                                                             test_size=0.3, random_state=42)

    # Logistic Regression
    lr_init = sklearn.linear_model.LogisticRegression()
    lr_param_grid = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    ## Validate over seasons
    gkf = GroupKFold(n_splits=len(train_Szn.unique()))

    ## Scale data to train set
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    # Save scaler
    s3_resource = boto3.resource('s3')
    pickle_byte_obj = pickle.dumps(scaler)
    s3_resource.Object('games-all', 'Models/Ensemble/game_probabilities_scaler').put(Body=pickle_byte_obj)

    ## Scale full dataframe
    full_df = all_game_probs.loc[:, feature_set]
    full_df = scaler.transform(full_df)
    full_df = pd.DataFrame(full_df, columns=feature_set)

    # LR1
    lr_cvsearch = sklearn.model_selection.GridSearchCV(lr_init, lr_param_grid, scoring='neg_log_loss', cv=10, verbose=0,
                                                       n_jobs=-1)
    lr_cvsearch.fit(train_X, train_Y)

    lr_cvsearch_out = model_complete_scoring(lr_cvsearch, 'games-all', 'Models/Ensemble/lr_model_cvsearch', test_X,
                                             test_Y, "proba")

    print("LR Grid Search Results: " + str(lr_cvsearch_out[:4]))

    # Logisitc model
    lr_randomsearch = sklearn.model_selection.RandomizedSearchCV(estimator=lr_init,
                                                                 param_distributions=lr_param_grid,
                                                                 cv=gkf,
                                                                 random_state=1, n_iter=10, verbose=0, n_jobs=-1)

    lr_randomsearch.fit(train_X, train_Y, groups=train_Szn)

    lr_randomsearch_out = model_complete_scoring(lr_randomsearch, 'games-all', 'Models/Ensemble/lr_model_randomsearch',
                                                 test_X, test_Y, "proba")

    print("LR Random Search Results: " + str(lr_randomsearch_out[:4]))

    lr_regularized = sklearn.linear_model.LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        , penalty='l2'
        , cv=10
        , scoring='neg_log_loss'
        , random_state=777
        , max_iter=10000
        , tol=10)

    lr_regularized.fit(train_X, train_Y)

    lr_regularized_out = model_complete_scoring(lr_regularized, 'games-all', 'Models/Ensemble/lr_model_regularized',
                                                test_X, test_Y, "proba")

    print("LR Regularized Results: " + str(lr_regularized_out[:4]))

    # predictions = pd.concat([
    #                test_Y.reset_index(drop=True),
    #                lr_cvsearch_out[4].reset_index(drop=True),
    #                lr_randomsearch_out[4].reset_index(drop=True),
    #                lr_regularized_out[4].reset_index(drop=True)
    #               ], axis = 1).fillna(method='ffill')

    # predictions.columns = ['home_win', 'lr1', 'lr2','lr3']

    ######
    ## Ensemble models
    ######
    ensembled_probs = pd.DataFrame()

    for model in ensemble_list:
        # Load models
        loaded_model = s3_model_object_load('games-all', 'Models/Ensemble/' + str(model))

        pred = pd.DataFrame(loaded_model.predict_proba(full_df)).iloc[:, 1]

        pred = pred.to_frame()
        pred.columns = [str(model)]

        ensembled_game_probs = pd.concat([ensembled_game_probs.reset_index(drop=True), pred], axis=1)

    return (ensembled_game_probs)

def output_goalie_prediction_probs(ensembled_game_probs):
    goalie_prediction_matrix = read_boto_s3('games-all', 'goalie_prediction_matrix.csv') \
        .groupby(['id', 'away_starter_id', 'home_starter_id']).first().reset_index()

    from datetime import date, timedelta
    ensembled_game_probs = ensembled_game_probs.loc[ensembled_game_probs.status != 'Final', :]

    ensembled_game_probs['home_win_probabilty'] = ensembled_game_probs \
        [['lr_model_regularized']] \
        .mean(axis=1) # 'lr_model_randomsearch', 'lr_model_cvsearch',

    possible_starters = read_boto_s3('games-all', 'possible-starters.csv')

    ensembled_game_probs2 = ensembled_game_probs \
        .merge(possible_starters[['starter_id', 'starter_name']].drop_duplicates() \
               .rename(index=str, columns={"starter_id": "home_starter_id", "starter_name": "home_starter"}),
               on=['home_starter_id'], how='left') \
        .merge(possible_starters[['starter_id', 'starter_name']].drop_duplicates() \
               .rename(index=str, columns={"starter_id": "away_starter_id", "starter_name": "away_starter"}),
               on=['away_starter_id'], how='left')

    ensembled_game_probs2[['away_starter', 'home_starter']] = ensembled_game_probs2[
        ['away_starter', 'home_starter']].fillna('Replacement-Level')

    ensembled_game_probs2['date'] = date.today()

    goalie_prediction_matrix = goalie_prediction_matrix \
        .append(ensembled_game_probs2[
                    ['date', 'id', 'season', 'home_team', 'away_team', 'away_starter_id', 'home_starter_id',
                     'away_starter', 'home_starter', 'home_win_probabilty']]) \
        .groupby(['id', 'away_starter_id', 'home_starter_id']).last().reset_index()

    goalie_prediction_matrix['home_ml'] = goalie_prediction_matrix.apply(
        lambda x: implied_pct_to_ml(x.home_win_probabilty), axis=1)
    goalie_prediction_matrix['away_ml'] = goalie_prediction_matrix.apply(
        lambda x: implied_pct_to_ml(1 - x.home_win_probabilty), axis=1)

    write_boto_s3(goalie_prediction_matrix, 'games-all', 'goalie_prediction_matrix.csv')

    return (goalie_prediction_matrix)


def output_goalie_matrix(goalie_prediction_matrix):

    # Todays games
    goalie_prediction_matrix2 = goalie_prediction_matrix \
                                    .loc[pd.to_datetime(goalie_prediction_matrix['date']).dt.date == date.today(), :]

    for i in (goalie_prediction_matrix2['id'].drop_duplicates()):
        goalie_prediction_matrix2 = goalie_prediction_matrix \
                                        .loc[goalie_prediction_matrix['id'] == i, :]
        # .loc[(goalie_prediction_matrix['away_starter_id'] > 1) & (goalie_prediction_matrix['home_starter_id'] > 1), :]

        goalie_prediction_plt = goalie_prediction_matrix2.pivot(index='home_starter', columns='away_starter',
                                                                values='home_win_probabilty')

        import numpy as np
        import seaborn as sns
        import matplotlib.pylab as plt

        sns.set(style="white")

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        ax = sns.heatmap(goalie_prediction_plt, cmap=cmap, annot=True, fmt='.1%', center=0.5)

        plt.title('Home Win Probability - ' + str(goalie_prediction_matrix2['away_team'].iloc[0]) + \
                  " @ " + str(goalie_prediction_matrix2['home_team'].iloc[0]) + " - " + \
                  str(goalie_prediction_matrix2['date'].iloc[0]))

        plt.xlabel('Away Starter')
        plt.ylabel('Home Starter')

        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)

        s3_resource = boto3.resource('s3')
        s3_resource.Object('games-all', 'ProbMatrix/' + \
                           str(goalie_prediction_matrix2['date'].iloc[0]) + "-" + \
                           str(goalie_prediction_matrix2['away_team'].iloc[0]) + "v" + \
                           str(goalie_prediction_matrix2['home_team'].iloc[0]) + "-" + \
                           str(goalie_prediction_matrix2['id'].iloc[0])).put(Body=img_data)

        plt.clf()