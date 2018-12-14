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


def yesterday_last_gameid():
    """
    Return last game ID from finished games
    """
    out = json_schedule.get_schedule(date.today() - timedelta(1),date.today()- timedelta(1))
    GameId_List = []

    for i in range(len(out['dates'][0]['games'])):
        GameId_List.append(out['dates'][0]['games'][i]['gamePk'])
    return(int(str(max(GameId_List))[5:]))

def read_boto_s3(bucket, file):
    """
    read file from s3 to pandas
    :return:
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=file)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return(df)

## Define elo functions
mean_elo = 1500
elo_width = 400


def past_game_starter(df, year):
    """
    :param df:
    :param year:
    :return:
    """
    starter_vars = ['Date', 'Game_Id', 'Away_Goalie_Id', 'Away_Goalie', 'Away_Team', 'Home_Goalie_Id',
                    'Home_Goalie', 'Period', 'Seconds_Elapsed']

    starter_information = df.loc[:, starter_vars] \
        .dropna(subset=['Away_Goalie_Id', 'Home_Goalie_Id', 'Date', 'Game_Id', 'Period', 'Seconds_Elapsed']) \
        .sort_values(['Date', 'Game_Id', 'Period', 'Seconds_Elapsed'], ascending=True) \
        .dropna() \
        .groupby(['Date', 'Game_Id']) \
        .head(1) \
        .rename(index=str, columns={"Home_Goalie_Id": "home_starter_id", "Away_Goalie_Id": "away_starter_id"})

    starter_information = starter_information.loc[:, ['Game_Id', 'home_starter_id', 'away_starter_id']].dropna()

    starter_information['id'] = ((year * 1000000) + starter_information['Game_Id'].astype(float)).astype(int)

    return (starter_information.loc[:, ['id', 'home_starter_id', 'away_starter_id']])


def update_starter_df(year, start_game_id, end_game_id, starter_df):
    """
    :param year:
    :param start_game_id:
    :param end_game_id:
    :param starter_df:
    :return:
    """
    games_list = json_schedule.get_dates(list(range(start_game_id + 1, int(str(year) + "0" + str(end_game_id)))))

    pbp = scrape_functions.scrape_list_of_games(games_list, False)
    df = pd.DataFrame(pbp[0], columns=['Game_Id', 'Date', 'Period', 'Event', 'Description', 'Time_Elapsed',
                                       'Seconds_Elapsed', 'Strength', 'Ev_Zone', 'Type', 'Ev_Team',
                                       'Home_Zone', 'Away_Team', 'Home_Team', 'p1_name', 'p1_ID', 'p2_name',
                                       'p2_ID', 'p3_name', 'p3_ID', 'awayPlayer1', 'awayPlayer1_id',
                                       'awayPlayer2', 'awayPlayer2_id', 'awayPlayer3', 'awayPlayer3_id',
                                       'awayPlayer4', 'awayPlayer4_id', 'awayPlayer5', 'awayPlayer5_id',
                                       'awayPlayer6', 'awayPlayer6_id', 'homePlayer1', 'homePlayer1_id',
                                       'homePlayer2', 'homePlayer2_id', 'homePlayer3', 'homePlayer3_id',
                                       'homePlayer4', 'homePlayer4_id', 'homePlayer5', 'homePlayer5_id',
                                       'homePlayer6', 'homePlayer6_id', 'Away_Players', 'Home_Players',
                                       'Away_Score', 'Home_Score', 'Away_Goalie', 'Away_Goalie_Id',
                                       'Home_Goalie', 'Home_Goalie_Id', 'xC', 'yC', 'Home_Coach',
                                       'Away_Coach'])

    new_starters = past_game_starter(df, year)

    return (new_starters)

def encode_data(df):
    """
    :param df: data with encoding issues
    :return df2: data without encoding issues
    """
    df.to_csv('df.csv', index=False)
    df2 = pd.read_csv("df.csv", encoding='latin-1')
    os.remove("df.csv")
    return (df2)

def write_boto_s3(df, bucket, filename):
    """
    write csv file to s3
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, filename).put(Body=csv_buffer.getvalue())
    print("S3 " + str(bucket) + "/" + str(filename) + " updated")

def expected_result(elo_a, elo_b):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))
    return expect_a


def update_elo(home_elo, away_elo, k_factor, home_win):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    if home_win == 1:
        expected_win = expected_result(home_elo, away_elo)
        change_in_elo = k_factor * (1 - expected_win)
        home_elo += change_in_elo
        away_elo -= change_in_elo
    else:
        expected_win = expected_result(away_elo, home_elo)
        change_in_elo = k_factor * (1 - expected_win)
        away_elo += change_in_elo
        home_elo -= change_in_elo

    return home_elo, away_elo


### Rolling Weighted Average Functions
def roll(df, window, **kwargs):
    """
    Rolling weighted calculations
    """
    roll_array = np.dstack([df.values[i:i + window, :] for i in range(len(df.index) - window + 1)]).T
    panel = pd.Panel(roll_array,
                     items=df.index[window - 1:],
                     major_axis=df.columns,
                     minor_axis=pd.Index(range(window), name='roll'))
    return panel.to_frame().unstack().T.groupby(level=0, **kwargs)


def weighted_mean_offset(x, w, window, offset_decay, var_offset):
    """
    :param x:
    :param w:
    :param window:
    :param offset_decay:
    :param var_offset: do we know the result at puck drop or not? no offset 1
    :return:
    """
    w_arr = np.array(w[-window:])
    if var_offset == 1:
        x_arr = np.array(x[-window-1:-1])
    else:
        x_arr = np.array(x[-window:])
    count_back = np.cumsum(w_arr[::-1])[::-1]
    count_back_decay = (offset_decay + np.max(count_back)) - count_back
    weight = count_back_decay / np.sum(count_back_decay)
    return (weight*x_arr).sum()


def team_day_games_fun(game_info_data, game_roster_similarity):
    ### Track game results
    team_day_games = pd.merge(game_info_data.loc[:,
                              ['id', 'home_team', 'away_team', 'game_start_est', 'home_goals', 'away_goals',
                               'home_scores', 'away_scores', 'shootout_final']],
                              game_roster_similarity, on=['id'], how='left')

    team_day_games['date'] = (
                pd.to_datetime(team_day_games['game_start_est'], utc=True) - pd.Timedelta(hours=3)).dt.date

    team_day_games['home_win'] = team_day_games.apply(
        lambda x: 1 if x.home_goals > x.away_goals or x.home_scores > x.away_scores
        else 0, axis=1)

    team_day_games['win_margin'] = abs(team_day_games['home_goals'] - team_day_games['away_goals'])
    team_day_games['shootout_final'] = team_day_games.apply(
        lambda x: 1 if x.shootout_final == True or x.shootout_final == 1 else 0, axis=1)

    return (team_day_games)


def roster_simiarity_fun(pre_data, post_data, team):
    """
    Calculate roster similarity carrying over from the prior game
    :param pre_data:
    :param post_data:
    :param team:
    :return:
    """
    # Subset to team
    pre_team_data = pre_data.loc[
                    ((pre_data['home_team'] == team) & (pre_data['team'] == "home")) |
                    ((pre_data['away_team'] == team) & (pre_data['team'] == "away")), :]
    # Keep only last game
    pre_team_data = pre_team_data.loc[pre_team_data['id'] == max(pre_team_data['id']), :]

    # Subset new games to team
    post_team_data = post_data.loc[
                     ((post_data['home_team'] == team) & (post_data['team'] == "home")) |
                     ((post_data['away_team'] == team) & (post_data['team'] == "away")), :]

    # Combine data
    team_df = pre_team_data.append(post_team_data)


    team_df['player_next_game'] = team_df.groupby(['fullName'])['id'].shift(-1)

    team_df = team_df.loc[:, ['player_next_game', 'id']].groupby('id').min().reset_index(). \
        rename(index=str, columns={"player_next_game": "team_next_game"}). \
        merge(team_df, on=['id'], how='inner')

    team_df['player_played'] = team_df.apply(lambda x: 1 if x.player_next_game == x.team_next_game else 0, axis=1)
    team_df['skater'] = team_df.apply(lambda x: 1 if x.Pos != "G" else 0, axis=1)

    team_df['TOImin'], team_df['TOIsec'] = team_df['timeOnIce'].str.split(':', 1).str

    team_df['TOI'] = ((60 * team_df['TOImin'].str.lstrip('0').fillna(0).apply(lambda x: 0 if x == '' else x)
                       .astype(int)) + team_df['TOIsec'].str[:2].str.lstrip('0').apply(lambda x: 0 if x == '' else x)
                      .astype(int)) / 60

    ## Skater Only
    skater_df = team_df.groupby('id')['TOI', 'player_played', 'skater']. \
        agg(lambda x: sum(x.player_played * x.TOI * x.skater) / sum(x.TOI * x.skater)
    if sum(x.player_played * x.TOI * x.skater) / sum(x.TOI * x.skater) > 0
    else sum(x.player_played * x.skater) / len(x.player_played * x.skater)). \
        rename(columns={"TOI": "skater_similarity"}).reset_index()

    ## All Roster, calculate total TOI playing in the next game (goalies included)
    roster_df = team_df.groupby('id')['TOI', 'player_played', 'skater'].agg(
        lambda x: sum(x.player_played * x.TOI) / sum(x.TOI)
        if sum(x.player_played * x.TOI) / sum(x.TOI) > 0
        else sum(x.player_played) / len(x.player_played)). \
        rename(columns={"TOI": "allroster_similarity"}).reset_index()

    ## Combine
    all_roster_df = pd.merge(roster_df[['id', 'allroster_similarity']], skater_df[['id', 'skater_similarity']],
                             on='id', how='left')

    ## Shift to next game to represent carryover of roster
    all_roster_df['skater_similarity'] = all_roster_df['skater_similarity'].shift(1).fillna(1)
    all_roster_df['allroster_similarity'] = all_roster_df['allroster_similarity'].shift(1).fillna(1)
    all_roster_df['team'] = team

    # Drop prior game
    all_roster_df = all_roster_df.loc[all_roster_df['id'] > max(pre_team_data['id']), :]

    return (all_roster_df)


def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def wcorr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


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
    s3_resource = resource('s3')
    pickle_byte_obj = pickle.dumps(model)
    s3_resource.Object(bucket, 'Models/' + str(filename)).put(Body=pickle_byte_obj)

    #pickle.dump(model, open('./GameModelData/Models/' + str(filename), 'wb'))

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


def process_games_ytd(szn,
                      last_game,
                      game_info_data,
                      game_roster_data,
                      team_day_ratings_lag,
                      arena_geocode):

    ## If first game of the season set to 20001
    if game_info_data.loc[game_info_data.season == int(str(szn) + str(szn + 1)), 'Game_Id'].tail(1).shape[0] > 0:
        prior_game = int(game_info_data.loc[game_info_data.season == int(str(szn) + str(szn + 1)), 'Game_Id'].tail(1))
    else:
        prior_game = 20000

    games = list(range(prior_game + 1, last_game + 1))

    # Init the dataframe
    game_info = pd.DataFrame()
    game_roster = pd.DataFrame()

    ## For each game
    for i in games:

        if i % 10 == 0:
            print("GameID:" + str(i))

        try:
            url = "https://statsapi.web.nhl.com/api/v1/game/" + str(szn) + "0" + str(i) + "/feed/live?site=en_nhl"

            # Read game
            data = requests.get(url).json()

            if data['gameData']['status']['detailedState'] == 'Final':

                game = pd.DataFrame(data['gameData']['game'], index=[0]) \
                    .rename(index=str, columns={"pk": "id", "type": "season_type"})

                game['Game_Id'] = game['id'].astype(str).str.slice(5)

                home_scores = pd.DataFrame(data['liveData']['boxscore']['teams']['home']['teamStats']['teamSkaterStats'],\
                                           index=[0]).add_prefix('home_')
                home_scores['home_team'] = data['gameData']['teams']['home']['triCode']
                home_scores['home_coach'] = data['liveData']['boxscore']['teams']['home']['coaches'][0]['person']['fullName']


                away_scores = pd.DataFrame(data['liveData']['boxscore']['teams']['away']['teamStats']['teamSkaterStats'],\
                                           index=[0]).add_prefix('away_')
                away_scores['away_team'] = data['gameData']['teams']['away']['triCode']
                away_scores['away_coach'] = data['liveData']['boxscore']['teams']['away']['coaches'][0]['person']['fullName']


                home_shootout = pd.DataFrame(data['liveData']['linescore']['shootoutInfo']['home'], index=[0])\
                    .add_prefix('home_')
                away_shootout = pd.DataFrame(data['liveData']['linescore']['shootoutInfo']['away'], index=[0])\
                    .add_prefix('away_')

                has_shootout = data['liveData']['linescore']['hasShootout']

                game_level_data = pd.concat([home_scores,\
                                              away_scores,\
                                              home_shootout,\
                                              away_shootout], axis=1).reset_index()

                game_data = pd.merge(game.reset_index(),\
                                     game_level_data.reset_index(),\
                                     left_index=True, right_index=True).drop(['index_x', 'index_y'], axis=1)

                game_data['shootout_final'] = has_shootout
                game_data['venue'] = data['gameData']['venue']['name']
                game_data['status'] = data['gameData']['status']['detailedState']

                game_data['city'] = data['gameData']['teams']['home']['venue']['city']
                game_data['city'] = game_data['city'].str.replace("é", "e")

                game_data['timeZone'] = data['gameData']['teams']['home']['venue']['timeZone']['tz']
                game_data['timeZone_Num'] = data['gameData']['teams']['home']['venue']['timeZone']['offset']

                game_data['game_start_utc'] = pd.to_datetime(data['gameData']['datetime']['dateTime'], utc=True)

                game_data['game_start_est'] = game_data['game_start_utc']\
                        + pd.Timedelta(hours=int(game_data['timeZone_Num']))

                #game_data['home_starter_id'] = data['liveData']['boxscore']['teams']['home']['goalies'][0]
                #game_data['away_starter_id'] = data['liveData']['boxscore']['teams']['away']['goalies'][0]

                # Referees for the game
                refs = []
                lines = []

                for i in range(len(data['liveData']['boxscore']['officials'])):
                    name = data['liveData']['boxscore']['officials'][i]['official']['fullName']
                    ref_type = data['liveData']['boxscore']['officials'][i]['officialType']
                    if ref_type == 'Referee':
                        refs = refs + [name]
                    else:
                        lines = lines + [name]

                game_data['Referees'] = str(refs)
                game_data['Linesmen'] = str(lines)

                game_info = game_info.append(game_data)

                ## Roster Data
                roster_data = pd.DataFrame()

                away_players = data['liveData']['boxscore']['teams']['away']['players']

                for i, val in enumerate(away_players):
                    player = away_players[val]

                    person = pd.DataFrame(player['person'], index=[0]). \
                                 loc[:, ['id', 'fullName', 'rosterStatus', 'shootsCatches']].reset_index()
                    position = pd.DataFrame(player['position'], index=[0]). \
                                   loc[:, ['code']].rename(index=str, columns={"code": "Pos"}).reset_index()

                    stats = pd.DataFrame()

                    try:
                        stats = pd.DataFrame(player['stats']['skaterStats'], index=[0]).reset_index()
                    except:
                        pass

                    try:
                        stats = pd.DataFrame(player['stats']['goalieStats'], index=[0]).reset_index()
                    except:
                        pass

                    player_all = pd.merge(person, position, left_index=True, right_index=True)  # .drop(['index_x'], axis=1)
                    player_all = pd.merge(player_all, stats, left_index=True, right_index=True)\
                        .drop(['index_x', 'index_y'], axis=1)

                    player_all['team'] = 'away'
                    player_all['player_id'] = val[2:]

                    roster_data = roster_data.append(player_all)

                home_players = data['liveData']['boxscore']['teams']['home']['players']

                for i, val in enumerate(home_players):
                    player = home_players[val]

                    person = pd.DataFrame(player['person'], index=[0]). \
                                 loc[:, ['id', 'fullName', 'rosterStatus', 'shootsCatches']].reset_index()
                    position = pd.DataFrame(player['position'], index=[0]). \
                                   loc[:, ['code']].rename(index=str, columns={"code": "Pos"}).reset_index()

                    stats = pd.DataFrame()

                    try:
                        stats = pd.DataFrame(player['stats']['skaterStats'], index=[0]).reset_index()
                    except:
                        pass

                    try:
                        stats = pd.DataFrame(player['stats']['goalieStats'], index=[0]).reset_index()
                    except:
                        pass

                    player_all = pd.merge(person, position, left_index=True, right_index=True)  # .drop(['index_x'], axis=1)
                    player_all = pd.merge(player_all, stats, left_index=True, right_index=True)\
                        .drop(['index_x', 'index_y'], axis=1)

                    player_all['team'] = 'home'
                    player_all['player_id'] = val[2:]

                    roster_data = roster_data.append(player_all)

                roster_data['id'] = data['gameData']['game']['pk']

                game_roster = game_roster.append(roster_data)

            else:
                game = pd.DataFrame(data['gameData']['game'], index=[0]).rename(index=str, columns={"pk": "id",
                                                                                                    "type": "season_type"})
                game['Game_Id'] = game['id'].astype(str).str.slice(5)

                home_team = pd.DataFrame(data['gameData']['teams']['home'], index=[0]).loc[:, ['triCode']].rename(
                    index=str, columns={"triCode": "home_team"})
                away_team = pd.DataFrame(data['gameData']['teams']['away'], index=[0]).loc[:, ['triCode']].rename(
                    index=str, columns={"triCode": "away_team"})

                game_level_data = pd.concat([home_team,
                                             away_team], axis=1).reset_index()

                game_data = pd.merge(game.reset_index(),
                                     game_level_data.reset_index(),
                                     left_index=True, right_index=True).drop(['index_x', 'index_y', 'level_0'], axis=1)

                game_data['venue'] = data['gameData']['venue']['name']

                game_data['city'] = data['gameData']['teams']['home']['venue']['city']
                game_data['city'] = game_data['city'].str.replace("é", "e")

                game_data['timeZone'] = data['gameData']['teams']['home']['venue']['timeZone']['tz']
                game_data['timeZone_Num'] = data['gameData']['teams']['home']['venue']['timeZone']['offset']

                game_data['game_start_utc'] = pd.to_datetime(data['gameData']['datetime']['dateTime'], utc=True)

                game_data['game_start_est'] = game_data['game_start_utc'] + pd.Timedelta(
                    hours=int(game_data['timeZone_Num']))

                game_data['status'] = data['gameData']['status']['detailedState']

                game_info = game_info.append(game_data)
            print(str(szn) + ", Game:" + i)
        except:
            continue

    ## Create xref for new data
    team_game_xref = game_info_data.append(game_info).loc[:, ["id", "season", "home_team", "away_team"]]

    ## Geocode/xwalk new data
    game_info = game_info.merge(arena_geocode, on='city', how='left').drop(['level_0'], axis=1)
    game_roster = game_roster.merge(team_game_xref, on=["id"])

    ## Correct some team information
    # Replace every occurrence of PHX with ARI
    game_info['home_team'] = game_info.apply(lambda x: x.home_team if x.home_team != 'PHX' else 'ARI', axis=1)
    game_info['away_team'] = game_info.apply(lambda x: x.away_team if x.away_team != 'PHX' else 'ARI', axis=1)
    # Replace every occurrence of ATL with WPG
    game_info['home_team'] = game_info.apply(lambda x: x.home_team if x.home_team != 'ATL' else 'WPG', axis=1)
    game_info['away_team'] = game_info.apply(lambda x: x.away_team if x.away_team != 'ATL' else 'WPG', axis=1)


    ## Game by game roster similarity df
    team_game_roster_similarity = pd.DataFrame()

    # Only teams with new games
    team_list = game_info['home_team'].append(game_info['away_team']).drop_duplicates()

    for team in team_list:
        # Calculate roster similarity
        all_roster_df = roster_simiarity_fun(game_roster_data, game_roster, team)
        ## Append to empty DF
        team_game_roster_similarity = team_game_roster_similarity.append(all_roster_df)


    ## Game-level skater similarity from last game
    venue_xwalk = pd.melt(team_game_xref.loc[:, ["id", "home_team", "away_team"]]
                          .rename(index=str, columns={"home_team": "home_skater_sim", "away_team": "away_skater_sim"}),
                          id_vars=['id'], value_vars=['home_skater_sim', 'away_skater_sim'], var_name="venue",
                          value_name='team')

    team_game_roster_similarity['id'] = team_game_roster_similarity['id'].apply(int)

    game_roster_similarity = pd.merge(team_game_roster_similarity, venue_xwalk,\
                                      on=['id', 'team'], how='left')\
        .pivot(index='id', columns='venue', values='skater_similarity').reset_index()

    game_roster_similarity = pd.merge(pd.merge(team_game_roster_similarity, venue_xwalk,
                                               on=['id', 'team'], how='left')\
                                      .pivot(index='id', columns='venue', values='allroster_similarity').reset_index()\
                                      .rename(index=str, columns={"home_skater_sim": "home_roster_sim",
                                                                  "away_skater_sim": "away_roster_sim"}),
                                      game_roster_similarity, on=['id'], how='left')

    print("Roster Similarity Completed")

    ## Append data (drop 'season' first for new xwalk column)
    game_info_data = game_info_data.append(game_info)
    game_roster_data = game_roster_data.append(game_roster)


    ## Start Elo Ratings
    ### Track game results
    team_day_games = team_day_games_fun(game_info, game_roster_similarity)

    ### Game dates in sample
    game_dates = team_day_games['date'].drop_duplicates()

    ### Loop through each date and update rating
    for index, i in game_dates.iteritems():

        ## Subset to date
        day_games = team_day_games.loc[team_day_games.date == i, :]

        ## Grab teams last rating
        day_elos = team_day_ratings_lag.groupby('team').last().reset_index()

        ## For each game calculate elos after
        for index, game in day_games.iterrows():

            ## Team skater similarity
            home_sim = game['home_skater_sim']
            away_sim = game['away_skater_sim']

            ## Game outcome
            home_win = game['home_win']
            shootout_final = game['shootout_final']

            ## Win margin
            win_margin = game['win_margin']

            metrics = ['elo_k4', 'elo_k4_wm4', 'elo_k4_wm4_SO2', 'elo_k4_wm2_SO2', 'elo_k8_wm8_SO2']

            game_ratings = pd.DataFrame({'team': [game['home_team'], game['away_team']],
                                         'date': i
                                         })

            for metric in metrics:

                if metric == 'elo_k4':
                    k = 4
                elif metric == 'elo_k4_wm4':
                    k = (4 + (4 * win_margin))
                elif metric == 'elo_k4_wm2_SO2':
                    k = (4 + (2 * win_margin) - (shootout_final * 2))
                elif metric == 'elo_k4_wm4_SO2':
                    k = (4 + (4 * win_margin) - (shootout_final * 2))
                elif metric == 'elo_k8_wm8_SO2':
                    k = (8 + (8 * win_margin) - (shootout_final * 4))

                ## Past ratings
                home_rating = day_elos[day_elos['team'] == game['home_team']][str(metric)].values[0];
                away_rating = day_elos[day_elos['team'] == game['away_team']][str(metric)].values[0];

                ## Outcomes
                new_home_rating, new_away_rating = update_elo(home_rating, away_rating, k, home_win)

                metric_ratings = pd.DataFrame({'team': [game['home_team'], game['away_team']],
                                               'date': i,
                                               str(metric): [new_home_rating, new_away_rating]

                                               })

                game_ratings = pd.merge(game_ratings, metric_ratings, on=['date', 'team'], how='left')

            team_day_ratings_lag = team_day_ratings_lag.append(game_ratings)

    print("Elo Metrics updated")

    ## Create datasets for both home and away team
    away_team_ratings = team_day_ratings_lag.copy()
    home_team_ratings = team_day_ratings_lag.copy()

    ## Label appropriately
    away_team_ratings.columns = ['away_' + str(col) if col != 'date' else col for col in away_team_ratings.columns]
    home_team_ratings.columns = ['home_' + str(col) if col != 'date' else col for col in home_team_ratings.columns]

    team_day_elos = pd.merge(pd.merge(team_day_games,
                                      home_team_ratings,
                                      on=['date', 'home_team'], how='left'),
                             away_team_ratings,
                             on=['date', 'away_team'], how='left')

    return (game_info_data, game_roster_data, team_day_elos, team_day_ratings_lag)


def goalie_game_features(game_info_data,
                    goalie_rolling_df,
                    goalie_features = ['wa_hours_rest', 'wa_travel_km', 'wa_PP_svPct', 'wa_EV_svPct', 'wa_svPct'],
                    weights = ['_w12_o48', '_w12_o24', '_w6_o24', '_w6_o48']):
    """
    Converts goalie-level data into game-starter level data, assigning to home and away files
    :param game_info_data: all games
    :param goalie_rolling_df: goalie-level data
    :param goalie_features: goalie-features to keep
    :param weights:
    :return:
    """
    goalie_rolling_df['id'] = goalie_rolling_df['id'].astype(int)
    goalie_rolling_df['player_id'] = goalie_rolling_df['player_id'].astype(int)

    game_info_data['id'] = game_info_data['id'].astype(int)
    game_info_data['home_starter_id'] = game_info_data['home_starter_id'].astype(int)
    game_info_data['away_starter_id'] = game_info_data['away_starter_id'].astype(int)

    goalie_features = ['wa_hours_rest', 'wa_travel_km', 'wa_PP_svPct', 'wa_EV_svPct', 'wa_svPct']
    weights = ['_w12_o48', '_w12_o24', '_w6_o24', '_w6_o48']
    all_features = [str(feat) + str(w) for feat in goalie_features for w in weights]

    #home_goalie_metrics, away_goalie_metrics = goalie_game_features(game_info_data, goalie_rolling_df)
    ## Find 80/20 values to impute
    p80_results = goalie_rolling_df.loc[:,
                  [str(feat) + str(w) for feat in ['wa_hours_rest', 'wa_travel_km'] for w in weights]] \
        .dropna(axis=0, how='any') \
        .quantile(q=0.8, axis=0)

    p20_results = goalie_rolling_df.loc[:,
                  [str(feat) + str(w) for feat in ['wa_PP_svPct', 'wa_EV_svPct', 'wa_svPct'] for w in weights]] \
        .dropna(axis=0, how='any') \
        .quantile(q=0.2, axis=0)

    ## Home start metrics
    home_goalie_metrics = game_info_data \
        .loc[:, ['id', 'home_starter_id']] \
        .merge(goalie_rolling_df.loc[:, ['id', 'player_id'] + all_features],
               left_on=['id', 'home_starter_id'],
               right_on=['id', 'player_id'],
               how='inner') \
        .fillna(value=p80_results.append(p20_results)) \
        .drop(['player_id'], axis=1)

    home_goalie_metrics.columns = ['id', 'home_starter_id'] + ['home_starter_' + str(feat) for feat in all_features]

    away_goalie_metrics = game_info_data \
        .loc[:, ['id', 'away_starter_id']] \
        .merge(goalie_rolling_df.loc[:, ['id', 'player_id'] + all_features],
               left_on=['id', 'away_starter_id'],
               right_on=['id', 'player_id'],
               how='inner') \
        .fillna(value=p80_results.append(p20_results)) \
        .drop(['player_id'], axis=1)

    away_goalie_metrics.columns = ['id', 'away_starter_id'] + ['away_starter_' + str(feat) for feat in all_features]

    return(home_goalie_metrics, away_goalie_metrics)


def team_game_features_ytd(szn, game_info_data, team_day_elos, team_game_features, goalie_rolling_df, model_df):
    """
    :param szn:
    :param game_info_data_geocode:
    :param team_day_elos:
    :param team_game_features:
    :param model_df:
    :return:
    """
    pd.options.mode.chained_assignment = None
    roll_features = ['game_GF', 'game_GA', 'game_SF', 'game_SA', 'game_PPGF', 'game_PKGA', 'game_PPAtt', 'game_PKAtt']
    windows = [8, 12]
    hour_offsets = [24, 48]

    ## Prior season metrics
    game_info_data = game_info_data.loc[round(game_info_data.season.astype(float) / 10000) >= (szn - 1), :]

    ## Append to current metrics
    team_game_features = team_game_features.loc[round(team_game_features.id / 1000000) < (szn - 1), :]

    team_list = game_info_data['home_team'].drop_duplicates()

    ## Loop through each team
    for team in team_list:

        team_df = game_info_data.loc[
                  (game_info_data['home_team'] == team) | (game_info_data['away_team'] == team), :] \
            .sort_values(['id'])

        ## Last game geolocation
        team_df['last_city_lat'] = team_df.groupby('season')['city_lat'].shift(1).fillna(method='bfill').fillna(method='ffill')
        team_df['last_city_long'] = team_df.groupby('season')['city_long'].shift(1).fillna(method='bfill').fillna(method='ffill')

        ## Travel distance
        team_df['travel_km'] = team_df.apply(lambda x: vincenty((x.city_lat, x.city_long), \
                                                                (x.last_city_lat, x.last_city_long)), axis=1) \
                                   .astype(str).str[:-3].astype(float)

        ## Last game date
        team_df['last_game_start_est'] = team_df['game_start_est'].shift(1)
        ## Hours from last game
        team_df['hours_rest'] = (pd.to_datetime(team_df.game_start_est, utc=True) - \
                                 pd.to_datetime(team_df.last_game_start_est, utc=True)). \
            astype('timedelta64[h]').fillna(method='bfill')

        team_df['team'] = team

        ## Team specific results
        team_df['game_GF'] = team_df.apply(lambda x: x.home_goals if x.home_team == team else x.away_goals, axis=1)
        team_df['game_GA'] = team_df.apply(lambda x: x.home_goals if x.home_team != team else x.away_goals, axis=1)

        team_df['game_SF'] = team_df.apply(lambda x: x.home_shots if x.home_team == team else x.away_shots, axis=1)
        team_df['game_SA'] = team_df.apply(lambda x: x.home_shots if x.home_team != team else x.away_shots, axis=1)

        team_df['game_PPGF'] = team_df.apply(
            lambda x: x.home_powerPlayGoals if x.home_team == team else x.away_powerPlayGoals, axis=1)
        team_df['game_PKGA'] = team_df.apply(
            lambda x: x.home_powerPlayGoals if x.home_team != team else x.away_powerPlayGoals, axis=1)

        team_df['game_PPAtt'] = team_df.apply(
            lambda x: x.home_powerPlayOpportunities if x.home_team == team else x.away_powerPlayOpportunities, axis=1)
        team_df['game_PKAtt'] = team_df.apply(
            lambda x: x.home_powerPlayOpportunities if x.home_team != team else x.away_powerPlayOpportunities, axis=1)

        ## Roll-up weighted sum of each metric
        for i in roll_features:
            for window in windows:
                for hour_offset in hour_offsets:
                    output = roll(team_df, window+1).apply(
                        lambda x: weighted_mean_offset(x[i], x.hours_rest, window, hour_offset, 1)) \
                        .rename("wa_" + str(i) + "_w" + str(window) + "_" + str(hour_offset) + "hr")

                    team_df = team_df.join(output)

        team_df = team_df.loc[:, ['id', 'home_team', 'away_team', 'team', 'travel_km', 'hours_rest'] \
                                 + ['wa_' + str(feat) + "_w" + str(window) + "_" + str(hour_offset) + "hr" \
                                    for feat in roll_features for window in windows for hour_offset in hour_offsets]]

        team_game_features = team_game_features.append(team_df)

    print("Team Season YTD Features updated")

    ## Split features into home and away dataframes again
    away_team_game_features = team_game_features.loc[team_game_features.away_team == team_game_features.team, :]
    home_team_game_features = team_game_features.loc[team_game_features.home_team == team_game_features.team, :]

    away_team_game_features.drop(['home_team', 'away_team'], axis=1, inplace=True)
    home_team_game_features.drop(['home_team', 'away_team'], axis=1, inplace=True)


    ## Re-name
    away_team_game_features.columns = ['away_' + str(col) if col != 'id' else col for col in
                                       away_team_game_features.columns]
    home_team_game_features.columns = ['home_' + str(col) if col != 'id' else col for col in
                                       home_team_game_features.columns]

    write_boto_s3(game_info_data, 'games-all', 'game_info_data_prior.csv')
    write_boto_s3(goalie_rolling_df, 'games-all', 'goalie_rolling_df_prior.csv')

    home_goalie_metrics, away_goalie_metrics = goalie_game_features(game_info_data, goalie_rolling_df)

    ## Remove games in 1997 where rolling average didn't have enough games
    model_df1 = team_day_elos\
                    .merge(home_team_game_features, on=['id', 'home_team'], how='left')\
                    .merge(away_team_game_features, on=['id', 'away_team'], how='left') \
                    .merge(game_info_data.loc[:,['id','home_starter_id','away_starter_id']], on=['id'], how='left')\
                    .merge(home_goalie_metrics, on=['id', 'home_starter_id'], how='left')\
                    .merge(away_goalie_metrics, on=['id', 'away_starter_id'], how='left') \
                    .drop(['away_scores', 'home_scores'], axis=1)

    model_df1['season'] = model_df1['id'].apply(str).str[:4].apply(int)

    print("New model data shape:" + str(model_df1.shape))

    print("Prior model data shape:" + str(model_df.shape))

    model_df = model_df.append(model_df1)

    ## Limit to post lockout
    model_df = model_df.loc[model_df.season > 2004, :]

    print("Modeling dataset created, shape: " + str(model_df.shape))

    return (model_df, team_game_features)



def model_df(model_df,
             windows=[8, 12],
             hour_offsets=[24, 48],
             elo_metrics=['elo_k4_wm4', 'elo_k4_wm4_SO2', 'elo_k4_wm2_SO2'],
             model_list=['lr_cvsearch', 'gnb_isotonic', 'rf_isotonic', 'mlp_isotonic', 'lr_model'],
             game_features=['game_GF', 'game_GA', 'game_SF', 'game_SA', 'game_PPGF', 'game_PKGA', 'game_PPAtt',
                            'game_PKAtt'],
             standard_features = ['hours_rest', 'travel_km', 'roster_sim', 'skater_sim']):
    """
    :param model_df:
    :param windows:
    :param hour_offsets:
    :param elo_metrics:
    :param model_list:
    :return:
    """

    all_game_probs = model_df.loc[:, ['home_win', 'id', 'season', 'shootout_final', 'win_margin']]

    ######
    # Splitting data into X and Y
    ######
    complete_feature_set = ["wa_" + str(feature) + "_w" + str(window) + "_" + str(offset) + "hr"
                            for feature in game_features
                            for window in windows
                            for offset in hour_offsets] + elo_metrics + standard_features

    complete_feature_list = [str(venue) + "_" + feat for venue in ['home', 'away'] for feat in complete_feature_set]

    train_X, test_X, train_Y, test_Y, train_Szn, test_Szn = train_test_split(model_df.loc[:, complete_feature_list],
                                                                             model_df.loc[:, 'home_win'],
                                                                             model_df.loc[:, 'season'],
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
    s3_resource.Object('games-all', 'Models/training_set_scaler').put(Body=pickle_byte_obj)

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
    rf_isotonic = CalibratedClassifierCV(rf, cv=5)  # , method='isotonic')

    # MultiLayer Perceptron
    mlp = MLPClassifier(learning_rate='adaptive',
                        hidden_layer_sizes=(10, 10, 10))
    mlp_isotonic = CalibratedClassifierCV(mlp, cv=5)  # , method='isotonic')

    ## Gaussian Naive-Bayes
    gnb = GaussianNB()
    gnb_isotonic = CalibratedClassifierCV(gnb, cv=5)  # , method='isotonic')

    # XGBoost
    xgb_init = XGBRegressor(objective='binary:logistic', alpha=10, n_estimators=10)

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

    for window in windows:
        for offset in hour_offsets:
            for metric in elo_metrics:

                print("metric: " + str(metric),
                      ", window: " + str(window),
                      ", offset: " + str(offset))

                # Set of features
                feature_set = ["wa_" + str(feature) + "_w" + str(window) + "_" + str(offset) + "hr"
                               for feature in game_features] + standard_features + [metric]

                feature_list = [str(venue) + "_" + feat for venue in ['home', 'away'] for feat in feature_set]

                full_scaled = full_df.loc[:, feature_list]

                # Model DF
                train_X0 = train_df.loc[:, feature_list].as_matrix()
                test_X0 = test_df.loc[:, feature_list].as_matrix()

                # LR1
                lr_cvsearch.fit(train_X0, train_Y)

                lr_cvsearch_out = model_complete_scoring(lr_cvsearch,'games-all',
                                                         'Models/lr_cvsearch_' + str(metric) + '_w' + str(window) + '_o' + str(
                                                             offset),
                                                         train_X0, train_Y, test_X0, test_Y, "proba")

                # Gaussian Naive-Bayes
                gnb_isotonic.fit(train_X0, train_Y)

                gnb_isotonic_out = model_complete_scoring(gnb_isotonic,'games-all',
                                                          'Models/gnb_isotonic_' + str(metric) + '_w' + str(
                                                              window) + '_o' + str(offset),
                                                          train_X0, train_Y, test_X0, test_Y, "proba")

                # MultiLayer Perceptron
                mlp_isotonic.fit(train_X0, train_Y)

                mlp_isotonic_out = model_complete_scoring(mlp_isotonic,'games-all',
                                                          'Models/mlp_isotonic_' + str(metric) + '_w' + str(
                                                              window) + '_o' + str(offset),
                                                          train_X0, train_Y, test_X0, test_Y, "proba")

                # Random Forest
                rf_isotonic.fit(train_X0, train_Y)

                rf_isotonic_out = model_complete_scoring(rf_isotonic,'games-all',
                                                         'Models/rf_isotonic_' + str(metric) + '_w' + str(window) + '_o' + str(
                                                             offset),
                                                         train_X0, train_Y, test_X0, test_Y, "proba")

                # XGBoost
                xgb_model.fit(train_X0, train_Y, groups=train_Szn)

                xgb_model_out = model_complete_scoring(xgb_model,'games-all',
                                                       'Models/xgb_model_' + str(metric) + '_w' + str(window) + '_o' + str(
                                                           offset),
                                                       train_X0, train_Y, test_X0, test_Y, "predict")

                # Logisitc model
                lr_model.fit(train_X0, train_Y, groups=train_Szn)

                lr_model_out = model_complete_scoring(lr_model,'games-all',
                                                      'Models/lr_model_' + str(metric) + '_w' + str(window) + '_o' + str(
                                                          offset),
                                                      train_X0, train_Y, test_X0, test_Y, "proba")

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
                    [round(blend_pred, 2).reset_index(drop=True), pd.Series(test_Y).reset_index(drop=True)], axis=1)
                calibration_df.columns = ['pred', 'home_win']
                calibration_agg = calibration_df.groupby('pred')['home_win'].agg(['mean', 'count']).reset_index()

                blend_calibration_corr = wcorr(np.array(calibration_agg['pred']),
                                               np.array(calibration_agg['mean']), np.array(calibration_agg['count']))

                cols = ["acc", "logloss", "brier_score", "cal_corr"]

                model_results = pd.concat(
                    [pd.DataFrame(np.array(
                        [metric, window, offset, blend_logloss, blend_brier_score, blend_calibration_corr]).reshape(1,
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
                     pd.DataFrame(np.array(lr_model_out[:4]).reshape(1, 4), columns=["lr_" + str(col) for col in cols]),
                     pd.DataFrame(np.array(xgb_model_out[:4]).reshape(1, 4),
                                  columns=["xgb_" + str(col) for col in cols])
                     ], axis=1)

                all_results = all_results.append(model_results)

                write_boto_s3(all_results, 'games-all', 'prediction_results_param.csv')

                test_predictions = pd.concat([pd.DataFrame(np.array([metric, window, offset]).reshape(1, 3)),
                                              test_Y.reset_index(drop=True),
                                              lr_cvsearch_out[4].reset_index(drop=True),
                                              gnb_isotonic_out[4].reset_index(drop=True),
                                              mlp_isotonic_out[4].reset_index(drop=True),
                                              rf_isotonic_out[4].reset_index(drop=True),
                                              lr_model_out[4].reset_index(drop=True),
                                              xgb_model_out[4].reset_index(drop=True),
                                              blend_pred.reset_index(drop=True)
                                              ], axis=1).fillna(method='ffill')

                test_predictions.columns = ['metric', 'window', 'offset', 'home_win', 'lrsearch_pred', 'gnb_pred',
                                            'mlp_pred', 'rf_pred', 'lr_pred', 'xgb_pred', 'blend_pred']

                test_predictions = test_predictions.append(test_predictions)

                write_boto_s3(test_predictions, 'games-all', 'test_results_param.csv')

                ######
                ## Ensemble models
                ######
                for model in model_list:
                    # Load models
                    s3 = boto3.resource('s3')
                    with io.BytesIO() as data:
                        s3.Bucket("games-all").download_fileobj('Models/' + \
                                    str(metric) + '_w' + str(window) + '_o' + str(offset), data)
                        data.seek(0)  # move back to the beginning after writing
                        m = pickle.load(data)

                    # Try
                    if model != 'xgb_model':
                        pred = pd.DataFrame(m.predict_proba(full_scaled)).iloc[:, 1]
                    else:
                        pred = pd.Series(m.predict(full_scaled))

                    pred = pred.to_frame()
                    pred.columns = [str(model) + "_" + str(metric) + '_w' + str(window) + '_o' + str(offset)]
                    all_game_probs = pd.concat([all_game_probs.reset_index(drop=True),
                                                pred], axis=1)

    return (all_game_probs)


def goalie_data(game_roster_data,
                game_info_data,
                season_start,
                goalie_rolling_df,
                windows=[6, 12],
                hour_offsets=[24, 48]):

    metrics = ['fullName', 'id', 'game_start_est', 'city_lat', 'city_long', 'TOI', 'season', 'Pos', 'timeOnIce',
               'saves', 'shots', 'decision', 'evenSaves', 'evenShotsAgainst', 'player_id', 'powerPlaySaves',
               'powerPlayShotsAgainst', 'shortHandedSaves', 'shortHandedShotsAgainst']
    output_metrics = ['fullName', 'player_id', 'TOI', 'id', 'game_start_est', 'home_team', 'away_team', 'hours_rest',
                      'travel_km', 'saves', 'shots', 'decision']

    control_feats = ['hours_rest', 'travel_km']
    result_feats = ['TOI', 'shots', 'saves', 'evenSaves', 'evenShotsAgainst', 'powerPlaySaves', 'powerPlayShotsAgainst']

    last_id = max(goalie_rolling_df['id'])

    game_info_new = game_info_data.loc[game_info_data.id.astype(float) > max(goalie_rolling_df['id']), :]

    print("New games to update for goalie data :" + str(game_info_new.shape))
    # Prior season metrics
    game_roster = game_roster_data.loc[round(game_roster_data.season.astype(float) / 10000) >= (season_start - 1) , :]
    game_roster = game_roster.loc[game_roster['Pos'] == "G", :]

    game_info = game_info_data.loc[round(game_info_data.season.astype(float) / 10000) >= (season_start - 1), :]

    ## Prep data
    game_info['game_start_est'] = pd.to_datetime(game_info['game_start_est'], utc=True)

    ## Prep data
    game_roster['TOImin'], game_roster['TOIsec'] = game_roster['timeOnIce'].str.split(':', 1).str

    game_roster['TOI'] = ((60 * game_roster['TOImin'].str.lstrip('0').fillna(0).apply(
        lambda x: 0 if x == '' else x).astype(int)) + \
                          game_roster['TOIsec'].str[:2].str.lstrip('0').apply(lambda x: 0 if x == '' else x).astype(
                              int)) / 60

    game_roster['season'] = game_roster['id'].astype(str).str[:4]
    game_roster = game_roster \
        .merge(game_info.loc[:, ['id', 'game_start_utc', 'game_start_est', 'city_lat', 'city_long']], on='id',
               how='inner') \
        .sort_values('game_start_est', ascending=True)

    # Split data
    goalie_data = game_roster.loc[game_roster.Pos == 'G', metrics]
    goalie_out = game_roster.loc[game_roster.Pos == 'G', output_metrics]

    goalie_df = pd.DataFrame()

    ## Unique goalie IDs
    goalie_game_totals = goalie_data.groupby(['player_id'])['id'].count()

    game_info_new = game_info_data.loc[game_info_data.id > max(goalie_rolling_df['id']), :]

    goalie_id = list(game_info_new['home_starter_id'].append(game_info_new['away_starter_id']).drop_duplicates())

    print(str(len(goalie_id)) + " goalies updating")

    for goalie in goalie_id:

        goalie_select = goalie_data.loc[goalie_data.player_id.astype(int) == goalie, :].reset_index(drop=True) \
            .sort_values('game_start_est', ascending=True).tail(50)
        goalie_select_out = goalie_out.loc[goalie_out.player_id.astype(int) == goalie, :].reset_index(drop=True) \
            .sort_values('game_start_est', ascending=True).tail(50)

        # print(str(goalie) + " ID - " + str(goalie_select_out.shape[0]) + " Games")

        if goalie_select_out.shape[0] > 1:
            try:
                ## Last game geolocation
                goalie_select['last_city_lat'] = goalie_select.groupby('season')['city_lat'].shift(1).fillna(
                    method='bfill').fillna(method='ffill')
                goalie_select['last_city_long'] = goalie_select.groupby('season')['city_long'].shift(1).fillna(
                    method='bfill').fillna(method='ffill')

                ## Travel distance
                goalie_select['travel_km'] = goalie_select.apply(lambda x: vincenty((x.city_lat, x.city_long), \
                                                                                    (
                                                                                    x.last_city_lat, x.last_city_long)),
                                                                 axis=1) \
                                                 .astype(str).str[:-3].astype(float)

                goalie_select['last_game_start_est'] = goalie_select['game_start_est'].shift(1)

                ## Hours from last game
                goalie_select['hours_rest'] = (pd.to_datetime(goalie_select.game_start_est, utc=True) - \
                                               pd.to_datetime(goalie_select.last_game_start_est, utc=True)). \
                    astype('timedelta64[h]').fillna(24 * 14)

                goalie_windows = [min(goalie_game_totals[goalie] - 1, win) for win in windows]

                for window in range(len(goalie_windows)):
                    for hour_offset in hour_offsets:
                        # For features known before game time
                        for i in control_feats:
                            output = roll(goalie_select, goalie_windows[window] + 1) \
                                .apply(lambda x: weighted_mean_offset(x[i], x.hours_rest, goalie_windows[window],
                                                                      hour_offset, 0)) \
                                .rename("wa_" + str(i) + "_w" + str(windows[window]) + "_o" + str(hour_offset))
                            goalie_select_out = goalie_select_out.join(output)

                        # For features as the result of the game
                        for i in result_feats:
                            output = roll(goalie_select, goalie_windows[window] + 1) \
                                .apply(lambda x: weighted_mean_offset(x[i], x.hours_rest, goalie_windows[window],
                                                                      hour_offset, 1)) \
                                .rename("wa_" + str(i) + "_w" + str(windows[window]) + "_o" + str(hour_offset))

                            goalie_select_out = goalie_select_out.join(output)

                goalie_select_out['travel_km'] = goalie_select['travel_km']
                goalie_select_out['hours_rest'] = goalie_select['hours_rest']
                goalie_select_out['days_rest'] = np.round(goalie_select_out['hours_rest'] / 24)
            except:
                continue

        # Append
        goalie_df = goalie_df.append(goalie_select_out)

    ## Subset to new goalie data
    goalie_df = goalie_df.loc[goalie_df['id'] > last_id, :]

    print(goalie_df.tail())

    goalie_rolling_df = goalie_rolling_df.append(goalie_df)

    weights = ['_w12_o48', '_w12_o24', '_w6_o24', '_w6_o48']

    for weight in weights:
        goalie_rolling_df['wa_PP_svPct' + str(weight)] = goalie_rolling_df['wa_powerPlaySaves' + str(weight)] / goalie_rolling_df['wa_powerPlayShotsAgainst' + str(weight)]
        goalie_rolling_df['wa_EV_svPct' + str(weight)] = goalie_rolling_df['wa_evenSaves' + str(weight)] / goalie_rolling_df['wa_evenShotsAgainst' + str(weight)]
        goalie_rolling_df['wa_svPct' + str(weight)] = goalie_rolling_df['wa_saves' + str(weight)] / goalie_rolling_df['wa_shots' + str(weight)]


    return (goalie_rolling_df)