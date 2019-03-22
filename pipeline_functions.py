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
import sklearn.metrics
import pickle
import boto3
import io
import os


def yesterday_last_gameid():
    """
    Return last game ID from finished games
    """
    out = json_schedule.get_schedule(date.today() - timedelta(1),date.today() - timedelta(1))
    GameId_List = []

    for i in range(len(out['dates'][0]['games'])):
        GameId_List.append(out['dates'][0]['games'][i]['gamePk'])
    return(int(str(max(GameId_List))[5:]))


def today_last_gameid():
    """
    Return last game ID from todays games
    """
    out = json_schedule.get_schedule(date.today(),date.today())
    GameId_List = []

    for i in range(len(out['dates'][0]['games'])):
        GameId_List.append(out['dates'][0]['games'][i]['gamePk'])
    return(int(str(max(GameId_List))[5:]))

def tomorrow_last_gameid():
    out = json_schedule.get_schedule(date.today() + timedelta(1),date.today() + timedelta(1))
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
    df = pd.read_csv(io.BytesIO(obj['Body'].read()), low_memory=False)
    return(df)

## Define elo functions
mean_elo = 1500
elo_width = 400

def encode_data(df, types = {}):
    """
    :param df: data with encoding issues
    :return df2: data without encoding issues
    """
    df.to_csv('df.csv', index=False)
    df2 = pd.read_csv("df.csv", encoding='latin-1', dtype = types)
    os.remove("df.csv")
    return (df2)


def goalie_starter_denote(data, team):
    goalie_list = data['liveData']['boxscore']['teams'][team]['goalies']

    if len(goalie_list) == 1:
        starter = goalie_list[0]
    else:
        ## Bring in all plays in string format
        allplays = str(data['liveData']['plays']['allPlays'])
        index = []
        for i in goalie_list:
            idx = allplays.find(str(i))
            idx = idx if idx > 1 else 99999999
            index = index + [idx]
            ## Return starter that shows up first in the all plays
        starter = goalie_list[np.argmin(index)]
    return (starter)


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
    elif home_win == 0:
        expected_win = expected_result(away_elo, home_elo)
        change_in_elo = k_factor * (1 - expected_win)
        away_elo += change_in_elo
        home_elo -= change_in_elo

    return home_elo, away_elo


def expand_future_games_possible_starters(game_info_data, num=2, include_replacement=True, remove_list=[]):
    possible_starters = read_boto_s3('games-all', 'possible-starters.csv')

    # Take last 2 non-injured goalies
    possible_starters = possible_starters \
                            .loc[~possible_starters.starter_name.isin(list(remove_list)), :] \
        .groupby(['team','starter_id'])['last_game_id'].agg(['max']) \
        .reset_index() \
        .sort_values('max', ascending=False) \
        .groupby(['team']).head(num)

    # Include a replacement level goalie?
    if include_replacement == True:
        replacement_df = pd.DataFrame({'team': possible_starters['team'].drop_duplicates(),
                                       'starter_id': 1,
                                       'starter_name': 'Replacement'
                                       })

        possible_starters = possible_starters.append(replacement_df)

    past_game_info = game_info_data.loc[game_info_data.status == 'Final', :]

    future_game_info = game_info_data.loc[game_info_data.status != 'Final', :]

    future_vars = ['Game_Id', 'id', 'away_team', 'home_team', 'city', 'city_lat', 'city_long', 'game_start_est',
                   'game_start_utc', 'season', 'season_type', 'status', 'timeZone', 'timeZone_Num', 'venue']

    # Away teams
    away_teams = future_game_info['away_team'].drop_duplicates()

    away_possible_starters = possible_starters.loc[
        possible_starters.team.isin(list(away_teams)), ['starter_id', 'team']] \
        .rename(index=str, columns={"team": "away_team", "starter_id": "away_starter_id"})

    # Home teams
    home_teams = future_game_info['home_team'].drop_duplicates()

    home_possible_starters = possible_starters.loc[
        possible_starters.team.isin(list(home_teams)), ['starter_id', 'team']] \
        .rename(index=str, columns={"team": "home_team", "starter_id": "home_starter_id"})

    future_game_info = future_game_info.loc[:, future_vars] \
        .merge(home_possible_starters, on=['home_team'], how='left') \
        .merge(away_possible_starters, on=['away_team'], how='left')

    future_game_info[['home_starter_id', 'away_starter_id']] = future_game_info[
        ['home_starter_id', 'away_starter_id']].astype(int)

    return (past_game_info.append(future_game_info))

def replacement_goalie_data(result_windows = [10, 20, 40]):

    replacement_history = pd.DataFrame({'player_id': [1] * max(result_windows),
                                       'hours_rest': [182.478873] * max(result_windows),
                                       'travel_km': [1277] * max(result_windows),
                                       'TOI': [53.292981] * max(result_windows),
                                       'saves': [21.862375] * max(result_windows),
                                       'evenSaves': [16.823615] * max(result_windows),
                                       'powerPlaySaves': [3.017339] * max(result_windows),
                                       'shots': [24.195463] * max(result_windows),
                                       'evenShotsAgainst': [18.419514] * max(result_windows),
                                       'powerPlayShotsAgainst': [3.531877] * max(result_windows)
                                       })

    return(replacement_history)


def replacement_xG_goalie_data(goalie_game_data, result_windows=[10, 20, 40]):

    median_shots = goalie_game_data.loc[:, ['SA', 'xG_total']].quantile(0.5, axis=0)

    replacement_lift = -0.01

    replacement_GA = median_shots[1] - (median_shots[0] * replacement_lift)

    replacement_history = pd.DataFrame({'SA_Goalie_Id': [1] * max(result_windows),
                                        'hours_rest': [182.478873] * max(result_windows),
                                        'Goal': [replacement_GA] * max(result_windows),
                                        'SA': [median_shots[0]] * max(result_windows),
                                        'xG_total': [median_shots[1]] * max(result_windows)
                                        })

    return (replacement_history)

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


def possible_starters(game_info_data, game_roster_data):
    game_info_data = game_info_data.loc[game_info_data.status == 'Final', :]

    game_roster_data = game_roster_data.loc[game_roster_data.Pos == 'G', :].drop_duplicates()

    teams_list = game_info_data['home_team'].append(game_info_data['away_team']).drop_duplicates()

    goalie_vars = ['id', 'season', 'home_starter_id', 'away_starter_id', 'home_team', 'away_team']
    possible_starters = pd.DataFrame()

    for team in teams_list:
        team_df = game_info_data \
            .loc[(game_info_data.home_team == team) | (game_info_data.away_team == team), goalie_vars] \
            .tail(25) \
            .merge(game_roster_data, on=['id', 'home_team', 'away_team'], how='left')

        team_df['starter_id'] = team_df.apply(lambda x: x.player_id if (x.home_team == team and x.team == 'home')
                                                                       or (
                                                                                   x.away_team == team and x.team == 'away') else '',
                                              axis=1)
        team_df['starter_name'] = team_df.apply(lambda x: x.fullName if (x.home_team == team and x.team == 'home')
                                                                        or (
                                                                                    x.away_team == team and x.team == 'away') else '',
                                                axis=1)
        team_df = team_df.loc[team_df['starter_name'] != '', :]

        # Keep 3 most recent goalies used in last 25 games
        goalies_used = team_df.groupby(['starter_id', 'starter_name'])['id'].agg(['count', 'max']) \
            .sort_values('max', ascending=False) \
            .rename(index=str, columns={"count": "game_count_last25", "max": "last_game_id"}) \
            .reset_index()

        goalies_used['team'] = team
        goalies_used['starter_id'] = goalies_used['starter_id'].astype(float).astype(int)

        possible_starters = possible_starters.append(goalies_used)

    return (possible_starters)


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
                      team_day_elos,
                      arena_geocode):

    ## If first game of the season set to 20001
    game_info_data = game_info_data.loc[game_info_data['status'] == 'Final', :]

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

                game_data['game_start_est'] = game_data['game_start_utc']  + pd.Timedelta(hours=int(game_data['timeZone_Num']))

                game_data['home_starter_id'] = goalie_starter_denote(data, 'home')
                game_data['away_starter_id'] = goalie_starter_denote(data, 'away')

                # Referees for the game
                refs = []
                lines = []

                for j in range(len(data['liveData']['boxscore']['officials'])):
                    name = data['liveData']['boxscore']['officials'][j]['official']['fullName']
                    ref_type = data['liveData']['boxscore']['officials'][j]['officialType']
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
    ## Drop previously scheduled games
    game_info_data = game_info_data.loc[game_info_data.status == 'Final',:].append(game_info)
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
            shootout_final = game['shootout_final']#.fillna(False)

            ## Win margin
            win_margin = game['win_margin']#.fillna(0)

            metrics = ['elo_k4', 'elo_k4_wm4', 'elo_k4_wm4_SO2', 'elo_k4_wm2_SO2', 'elo_k8_wm8_SO2']

            game_ratings = pd.DataFrame({'team': [game['home_team'], game['away_team']],
                                         'date': i
                                         })

            for metric in metrics:

                if (metric == 'elo_k4') and (win_margin > -1):
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

    # Fill elo scores to today
    team_day_ratings_lag2 = pd.DataFrame()
    for team in team_list:
        team_data = team_day_ratings_lag.loc[team_day_ratings_lag.team == team, :] \
            .append(pd.DataFrame({'date': [date.today()]})).fillna(method='ffill')
        team_day_ratings_lag2 = team_day_ratings_lag2.append(team_data)

    ## Create datasets for both home and away team
    away_team_ratings = team_day_ratings_lag2.copy()
    home_team_ratings = team_day_ratings_lag2.copy()

    ## Label appropriately
    away_team_ratings.columns = ['away_' + str(col) if col != 'date' else col for col in away_team_ratings.columns]
    home_team_ratings.columns = ['home_' + str(col) if col != 'date' else col for col in home_team_ratings.columns]

    team_day_elos_new = pd.merge(pd.merge(team_day_games,
                            home_team_ratings,
                             on=['date', 'home_team'], how='left'),
                             away_team_ratings,
                             on=['date', 'away_team'], how='left').drop_duplicates()

    team_day_elos = team_day_elos.append(team_day_elos_new)

    team_day_elos['date'] = pd.to_datetime(team_day_elos['date']).dt.date

    return (game_info_data, game_roster_data, team_day_elos, team_day_ratings_lag)


def goalie_game_features(game_info_data,
                         goalie_rolling_df,
                         control_feats=['hours_rest', 'travel_km'],
                         result_weights=['_w10_o48', '_w20_o48', '_w40_o48', '_w10_o168', '_w20_o168', '_w40_o168'],
                         control_weights=['_w10_o24', '_w10_o48']):
    """
    Converts goalie-level data into game-starter level data, assigning to home and away files
    :param game_info_data: all games
    :param goalie_rolling_df: goalie-level data
    :param goalie_features: goalie-features to keep
    :param weights:
    :return:
    """

    for weight in result_weights:
        goalie_rolling_df['wa_PP_svPct' + str(weight)] = goalie_rolling_df['wa_powerPlaySaves' + str(weight)] / \
                                                         goalie_rolling_df['wa_powerPlayShotsAgainst' + str(weight)]
        goalie_rolling_df['wa_EV_svPct' + str(weight)] = goalie_rolling_df['wa_evenSaves' + str(weight)] / \
                                                         goalie_rolling_df['wa_evenShotsAgainst' + str(weight)]
        goalie_rolling_df['wa_svPct' + str(weight)] = goalie_rolling_df['wa_saves' + str(weight)] / goalie_rolling_df[
            'wa_shots' + str(weight)]

    goalie_rolling_df['id'] = goalie_rolling_df['id'].astype(int)
    goalie_rolling_df['player_id'] = goalie_rolling_df['player_id'].astype(int)

    game_info_data['id'] = game_info_data['id'].astype(int)
    game_info_data['home_starter_id'] = game_info_data['home_starter_id'].astype(int)
    game_info_data['away_starter_id'] = game_info_data['away_starter_id'].astype(int)

    # goalie_features = ['wa_hours_rest', 'wa_travel_km', 'wa_PP_svPct', 'wa_EV_svPct', 'wa_svPct']

    result_features = [str(feat) + str(w) for feat in ['wa_PP_svPct', 'wa_EV_svPct', 'wa_svPct'] for w in
                       result_weights]
    control_features = [str(feat) + str(w) for feat in ['wa_hours_rest', 'wa_travel_km'] for w in control_weights]

    # home_goalie_metrics, away_goalie_metrics = goalie_game_features(game_info_data, goalie_rolling_df)
    ## Find 80/20 values to impute
    p50_results = goalie_rolling_df.loc[:,
                  [str(feat) + str(w) for feat in ['wa_hours_rest', 'wa_travel_km'] for w in control_weights]
                        + control_feats] \
        .dropna(axis=0, how='any') \
        .quantile(q=0.5, axis=0)

    p10_results = goalie_rolling_df.loc[:,
                  [str(feat) + str(w) for feat in ['wa_PP_svPct', 'wa_EV_svPct', 'wa_svPct'] for w in result_weights]] \
        .dropna(axis=0, how='any') \
        .quantile(q=0.1, axis=0)

    ## Home start metrics
    home_goalie_metrics = game_info_data \
                              .loc[:, ['id', 'home_starter_id']] \
        .drop_duplicates() \
        .merge(goalie_rolling_df.loc[:, ['id', 'player_id'] + control_features + result_features + control_feats],
               left_on=['id', 'home_starter_id'],
               right_on=['id', 'player_id'],
               how='left') \
        .fillna(value=p50_results.append(p10_results)) \
        .drop(['player_id'], axis=1)

    home_goalie_metrics.columns = ['id', 'home_starter_id'] + ['home_starter_' + str(feat) for feat in
                                                               control_features + result_features + control_feats]

    away_goalie_metrics = game_info_data \
                              .loc[:, ['id', 'away_starter_id']] \
        .drop_duplicates() \
        .merge(goalie_rolling_df.loc[:, ['id', 'player_id'] + control_features + result_features + control_feats],
               left_on=['id', 'away_starter_id'],
               right_on=['id', 'player_id'],
               how='left') \
        .fillna(value=p50_results.append(p10_results)) \
        .drop(['player_id'], axis=1)

    away_goalie_metrics.columns = ['id', 'away_starter_id'] + ['away_starter_' + str(feat) for feat in
                                                               control_features + result_features + control_feats]

    return (home_goalie_metrics, away_goalie_metrics)

def team_game_features_ytd(szn,
                           game_info_data,
                           team_day_elos,
                           score_adjusted_game_data,
                           team_shooting_info,
                           team_game_features,
                           goalie_rolling_df,
                           model_df,
                           remove_list):
    """
    :param szn:
    :param game_info_data_geocode:
    :param team_day_elos:
    :param team_game_features:
    :param model_df:
    :return:
    """
    pd.options.mode.chained_assignment = None
    result_features = ['game_skater_sim','game_GF','game_GA','game_SF',
                      'game_SA','game_PPGF','game_PKGA','game_PPAtt','game_PKAtt',
                      'game_SF_adj','game_SA_adj','game_xGF_adj','game_xGA_adj',
                      'game_SOGF_rate','game_SOGA_rate','game_FwdF_share','game_FwdA_share']
    control_features=['hours_rest', 'travel_km']
    result_windows = [10, 20, 40]
    result_offsets = [48, 168]
    control_windows = [10]
    control_offsets = [48]

    ## Prior season metrics
    game_info_data = game_info_data.loc[round(game_info_data.season.astype(float) / 10000) >= (szn - 1), :]

    ## Append to current metrics
    #team_game_features = team_game_features.loc[round(team_game_features.id / 1000000) < szn, :]
    team_game_features_new = pd.DataFrame()

    print("Prior team game features shape: " + str(team_game_features.shape))
    team_list = game_info_data['home_team'].drop_duplicates()

    ## Loop through each team
    for team in team_list:
        print(team)

        team_df = game_info_data.loc[(game_info_data['home_team'] == team) | (game_info_data['away_team'] == team), : ]\
                .sort_values(['game_start_est'])\
                .merge(team_day_elos.loc[:,['id','away_skater_sim','home_skater_sim']], on = 'id', how = 'left')\
                .merge(score_adjusted_game_data.loc[:,['id','Duration_60','away_SF_adj','home_SF_adj','away_xGF_adj','home_xGF_adj']], on = 'id', how = 'left')\
                .merge(team_shooting_info.loc[:,['id','home_SOG_rate','away_SOG_rate','away_F_shot_share','home_F_shot_share']], on = 'id', how = 'left')

        ## Last game geolocation
        team_df['last_city_lat'] = team_df.groupby('season')['city_lat'].shift(1).fillna(method='bfill')
        team_df['last_city_long'] = team_df.groupby('season')['city_long'].shift(1).fillna(method='bfill')

        ## Travel distance
        team_df['travel_km'] = team_df.apply(lambda x: vincenty((x.city_lat, x.city_long), \
                                                                (x.last_city_lat,  x.last_city_long)), axis = 1)\
                                    .astype(str).str[:-3].astype(float)

        ## Last game date
        team_df['last_game_start_est'] = team_df['game_start_est'].shift(1)
        ## Hours from last game
        team_df['hours_rest'] = (pd.to_timedelta(pd.to_datetime(team_df.game_start_est) - pd.to_datetime(team_df.last_game_start_est)) \
                                    / np.timedelta64(1, 'h')).fillna(24*7)
        team_df['team'] = team

        ## Team specific results
        team_df['game_GF'] = team_df.apply(lambda x: x.home_goals if x.home_team == team else x.away_goals, axis = 1)
        team_df['game_GA'] = team_df.apply(lambda x: x.home_goals if x.home_team != team else x.away_goals, axis = 1)

        team_df['game_SF'] = team_df.apply(lambda x: x.home_shots if x.home_team == team else x.away_shots, axis = 1)
        team_df['game_SA'] = team_df.apply(lambda x: x.home_shots if x.home_team != team else x.away_shots, axis = 1)

        team_df['game_PPGF'] = team_df.apply(lambda x: x.home_powerPlayGoals if x.home_team == team else x.away_powerPlayGoals, axis = 1)
        team_df['game_PKGA'] = team_df.apply(lambda x: x.home_powerPlayGoals if x.home_team != team else x.away_powerPlayGoals, axis = 1)

        team_df['game_PPAtt'] = team_df.apply(lambda x: x.home_powerPlayOpportunities if x.home_team == team else x.away_powerPlayOpportunities, axis = 1)
        team_df['game_PKAtt'] = team_df.apply(lambda x: x.home_powerPlayOpportunities if x.home_team != team else x.away_powerPlayOpportunities, axis = 1)

        team_df['game_SF_adj'] = team_df.apply(lambda x: x.home_SF_adj if x.home_team == team else x.away_SF_adj, axis = 1)
        team_df['game_SA_adj'] = team_df.apply(lambda x: x.home_SF_adj if x.home_team != team else x.away_SF_adj, axis = 1)

        team_df['game_xGF_adj'] = team_df.apply(lambda x: x.home_xGF_adj if x.home_team == team else x.away_xGF_adj, axis = 1)
        team_df['game_xGA_adj'] = team_df.apply(lambda x: x.home_xGF_adj if x.home_team != team else x.away_xGF_adj, axis = 1)

        team_df['game_SOGF_rate'] = team_df.apply(lambda x: x.home_SOG_rate if x.home_team == team else x.away_SOG_rate, axis = 1)
        team_df['game_SOGA_rate'] = team_df.apply(lambda x: x.home_SOG_rate if x.home_team != team else x.away_SOG_rate, axis = 1)

        team_df['game_FwdF_share'] = team_df.apply(lambda x: x.home_F_shot_share if x.home_team == team else x.away_F_shot_share, axis = 1)
        team_df['game_FwdA_share'] = team_df.apply(lambda x: x.home_F_shot_share if x.home_team != team else x.away_F_shot_share, axis = 1)

        team_df['game_skater_sim'] = team_df.apply(lambda x: x.home_skater_sim if x.home_team == team else x.away_skater_sim, axis = 1)\
            .fillna(method='ffill')

        ## Normalize per 60 team metrics
        count_features = ['game_GF', 'game_GA', 'game_SF',
                          'game_SA', 'game_PPGF', 'game_PKGA', 'game_PPAtt', 'game_PKAtt',
                          'game_SF_adj', 'game_SA_adj', 'game_xGF_adj', 'game_xGA_adj']

        team_df['Duration_60'] = team_df['Duration_60'].fillna(3600)

        for feat in count_features:
            team_df[feat] = team_df[feat] / (team_df['Duration_60'] / 3600)

        ## Impute pre-2005 features for rolling
        pbp_metrics = ['game_SF_adj', 'game_SA_adj', 'game_xGF_adj', 'game_xGA_adj']
        # team_df.fillna(team_df[pbp_metrics].mean(), inplace=True)
        cond = (team_df['status'] == 'Final')
        team_df.loc[cond, pbp_metrics] = team_df.loc[cond, pbp_metrics].fillna(team_df.loc[cond, pbp_metrics].median())

        # Future game flag
        team_df['last_game_status'] = team_df['status'].shift().fillna(method='bfill')

        future_df = team_df.loc[team_df['last_game_status'] != 'Final', :]
        team_df = team_df.loc[team_df['last_game_status'] == 'Final', :]

        # Control features
        for window in control_windows:
            for hour_offset in control_offsets:
                # For features known before game time
                for i in control_features:
                    output = roll(team_df, window) \
                        .apply(
                        lambda x: weighted_mean_offset(x[i], x.hours_rest, window,
                                                       hour_offset, 0)) \
                        .rename("wa_" + str(i) + "_w" + str(window) + "_o" + str(hour_offset))

                    team_df = team_df.join(output)

        # Results features
        for window in result_windows:
            for hour_offset in result_offsets:
                # For features as the result of the game
                for i in result_features:
                    output = roll(team_df, window + 1) \
                        .apply(lambda x: weighted_mean_offset(x[i], x.hours_rest, window,
                                                              hour_offset, 1)) \
                        .rename("wa_" + str(i) + "_w" + str(window) + "_o" + str(hour_offset))

                    team_df = team_df.join(output)

        # Append future games before dropping down
        team_df = team_df.append(future_df).fillna(method='ffill')

        # Limit to useful features
        team_df = team_df \
                      .sort_values('game_start_est') \
                      .loc[:,['id','home_team','away_team','team','travel_km','hours_rest']\
                              + ['wa_' + str(feat) + "_w" + str(window) + "_o" + str(hour_offset) \
                                 for feat in control_features
                                 for window in control_windows
                                 for hour_offset in control_offsets] \
                              + ['wa_' + str(feat) + "_w" + str(window) + "_o" + str(hour_offset) \
                                 for feat in result_features
                                 for window in result_windows
                                 for hour_offset in result_offsets]]\
                      .fillna(method='ffill')

        team_game_features_new = team_game_features_new.append(team_df)

    team_game_features_new = team_game_features_new.loc[team_game_features_new['id'] > max(team_game_features['id']), :]

    print("New team game features shape: " + str(team_game_features_new.shape))
    print("Old team game features shape: " + str(team_game_features.shape))

    team_game_features = team_game_features.append(team_game_features_new)
    print("Updated team game features shape: " + str(team_game_features.shape))

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

    ## Only update new data
    model_df = model_df.loc[model_df.status == 'Final',:].dropna(axis=0)
    last_id = max(model_df['id'])

    game_info_update = game_info_data.loc[game_info_data.id.astype(float) > last_id, :]
    team_day_elos = team_day_elos.loc[team_day_elos.id.astype(float) > last_id, :]

    game_info_update = expand_future_games_possible_starters(game_info_update,2,True,remove_list)

    home_goalie_metrics, away_goalie_metrics = goalie_xG_game_features(game_info_update, goalie_rolling_df)
    #home_goalie_metrics, away_goalie_metrics = goalie_game_features(game_info_update, goalie_rolling_df)

    ## Remove games in 1997 where rolling average didn't have enough games
    model_df1 = team_day_elos\
                    .merge(home_team_game_features, on=['id','home_team'], how='left')\
                    .merge(away_team_game_features, on=['id','away_team'], how='left') \
                    .merge(game_info_update.loc[:,['id','status','home_starter_id','away_starter_id']], on=['id'], how='left')\
                    .merge(home_goalie_metrics, on=['id','home_starter_id'], how='left')\
                    .merge(away_goalie_metrics, on=['id','away_starter_id'], how='left') \
                    .drop(['away_scores', 'home_scores'], axis=1)

    model_df1['season'] = model_df1['id'].apply(str).str[:4].apply(int)
    model_df1['date'] = (
                pd.to_datetime(model_df1['game_start_est'], utc=True) - pd.Timedelta(hours=3)).dt.date

    model_df1 = df_feature_transformer(model_df1)

    print("New model data shape:" + str(model_df1.shape))

    model_df = model_df.append(model_df1)
    model_df['season'] = model_df['id'].apply(str).str[:4].apply(int)

    ## Limit to post lockout
    model_df = model_df.loc[model_df.season > 2004, :]

    print("Modeling dataset created, shape: " + str(model_df.shape))

    return (model_df, team_game_features)

def goalie_data(game_roster_data,
                game_info_data,
                season_start,
                goalie_rolling_df):

    control_windows = [10]
    control_offsets = [24, 48]
    result_windows = [10, 20, 40]
    result_offsets = [48, 168]


    metrics = ['fullName', 'id', 'game_start_est', 'city_lat', 'city_long', 'TOI', 'season', 'Pos', 'timeOnIce',
               'saves', 'shots', 'decision', 'evenSaves', 'evenShotsAgainst', 'player_id', 'powerPlaySaves',
               'powerPlayShotsAgainst', 'shortHandedSaves', 'shortHandedShotsAgainst']
    output_metrics = ['fullName', 'player_id', 'TOI', 'id', 'game_start_est', 'home_team', 'away_team', 'hours_rest',
                      'travel_km', 'saves', 'shots', 'decision']

    control_feats = ['hours_rest', 'travel_km']
    result_feats = ['TOI', 'shots', 'saves', 'evenSaves', 'evenShotsAgainst', 'powerPlaySaves', 'powerPlayShotsAgainst']

    last_id = max(goalie_rolling_df['id'])

    ## Prep data
    game_info_data['game_start_est'] = pd.to_datetime(game_info_data['game_start_est'], utc=True)

    game_info_update = game_info_data.loc[(game_info_data['status'] == 'Final') &
                                          (game_info_data.id.astype(float) > last_id), :]

    # Prior season metrics
    game_roster = game_roster_data.loc[round(game_roster_data.season.astype(float) / 10000) >= (season_start - 1), :]
    game_roster = game_roster.loc[game_roster['Pos'] == "G", :]
        #.merge(game_info_update, on = ['id'], how = 'inner')


    ## Find future games too
    game_info_future = game_info_data.loc[game_info_data['status'] != 'Final', :]

    game_info_future = expand_future_games_possible_starters(game_info_future, 2, False)

    game_info_update = game_info_update.append(game_info_future)

    game_info_update[['away_starter_id', 'home_starter_id']] = game_info_update[
        ['away_starter_id', 'home_starter_id']].astype(float).astype(int)

    print("New games to update for goalie data :" + str(game_info_update.shape))

    ## Prep data
    game_roster['TOImin'], game_roster['TOIsec'] = game_roster['timeOnIce'].str.split(':', 1).str

    game_roster['TOI'] = ((60 * game_roster['TOImin'].str.lstrip('0').fillna(0).apply(
        lambda x: 0 if x == '' else x).astype(int)) + \
                          game_roster['TOIsec'].str[:2].str.lstrip('0').apply(lambda x: 0 if x == '' else x).astype(
                              int)) / 60

    game_info = game_info_data.loc[round(game_info_data.season.astype(float) / 10000) >= (season_start - 1), :]

    game_info_future2 = pd.melt(game_info_future.loc[:,
                                ['id', 'game_start_utc', 'game_start_est', 'city_lat', 'home_team', 'away_team',
                                 'city_long', 'away_starter_id', 'home_starter_id']],
                                id_vars=['id', 'game_start_utc', 'game_start_est', 'home_team', 'away_team', 'city_lat',
                                         'city_long'],
                                value_vars=['away_starter_id', 'home_starter_id'],
                                var_name='season',
                                value_name='player_id') \
        .merge(game_roster.loc[:, ['player_id', 'fullName', 'Pos']].drop_duplicates(), on=['player_id'], how='left') \
        .drop_duplicates()

    game_info_future2['player_id'] = game_info_future2['player_id'].astype(int)

    ## Join completed games and then append new goalie combinations
    game_roster = game_info.loc[:, ['id', 'game_start_utc', 'game_start_est', 'city_lat', 'city_long']] \
        .merge(game_roster, on=['id'], how='inner') \
        .append(game_info_future2) \
        .sort_values('id', ascending=True)

    game_roster['season'] = game_roster['id'].astype(str).str[:4]

    # Split data
    goalie_data = game_roster.loc[:, metrics]
    goalie_out = game_roster.loc[:, output_metrics]

    goalie_df = pd.DataFrame()

    goalie_id = list(
        game_info_update['home_starter_id'].append(game_info_update['away_starter_id']).drop_duplicates().dropna())

    print(str(len(goalie_id)) + " goalies updating")

    for goalie in goalie_id:

        goalie_select = goalie_data.loc[goalie_data.player_id.astype(int) == goalie, :].reset_index(drop=True) \
            .sort_values('game_start_est', ascending=True).tail(50)
        goalie_select_out = goalie_out.loc[goalie_out.player_id.astype(int) == goalie, :].reset_index(drop=True) \
            .sort_values('game_start_est', ascending=True).tail(50)

        print(str(goalie) + " ID - " + str(goalie_select_out.shape[0]) + " Games")

        goalie_select_out['game_count'] = goalie_select_out.groupby('player_id').cumcount() + 1
        goalie_select['game_index'] = goalie_select.groupby('player_id').cumcount()

        if goalie_select_out.shape[0] > 0:
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
                    astype('timedelta64[h]').fillna(24 * 14).astype(int)

                goalie_select['hours_rest'] = goalie_select['hours_rest'].apply(lambda x:(24*14) if x > (24*14) else x)

            except:
                continue

            goalie_select_out['travel_km'] = goalie_select['travel_km'].fillna(1277)
            goalie_select_out['hours_rest'] = goalie_select['hours_rest'].fillna(24 * 14)
            goalie_select_out['days_rest'] = np.round(goalie_select_out['hours_rest'] / 24)

            # If too few games append replacement level data on
            if goalie_select_out.shape[0] < 50:

                ## Append replacement data to start
                goalie_select = replacement_goalie_data(result_windows) \
                    .append(goalie_select, ignore_index=True)

                goalie_select_out = replacement_goalie_data(result_windows)[
                    ['hours_rest', 'player_id', 'travel_km']] \
                    .append(goalie_select_out, ignore_index=True)

            goalie_select = goalie_select.fillna(0)

            for window in range(len(control_windows)):
                for hour_offset in control_offsets:
                    # For features known before game time
                    for i in control_feats:
                        output = roll(goalie_select, control_windows[window] + 1) \
                            .apply(
                            lambda x: weighted_mean_offset(x[i], x.hours_rest, control_windows[window],
                                                           hour_offset, 0)) \
                            .rename("wa_" + str(i) + "_w" + str(control_windows[window]) + "_o" + str(hour_offset))

                        goalie_select_out = goalie_select_out.join(output)

            for window in range(len(result_windows)):
                for hour_offset in result_offsets:
                    # For features as the result of the game
                    for i in result_feats:
                        output = roll(goalie_select, result_windows[window] + 1) \
                            .apply(lambda x: weighted_mean_offset(x[i], x.hours_rest, result_windows[window],
                                                                  hour_offset, 1)) \
                            .rename("wa_" + str(i) + "_w" + str(result_windows[window]) + "_o" + str(hour_offset))

                        goalie_select_out = goalie_select_out.join(output)

        goalie_select_out = goalie_select_out.loc[goalie_select_out.player_id.astype(int) > 2, :]

        # Append
        goalie_df = goalie_df.append(goalie_select_out)

    ## Subset to new goalie data
    goalie_df = goalie_df.loc[goalie_df.id.astype(int) > last_id, :]

    goalie_rolling_df = goalie_rolling_df.append(goalie_df)

    weights = ['_w10_o48', '_w20_o48', '_w40_o48', '_w10_o168', '_w20_o168', '_w40_o168']

    for weight in weights:
        goalie_rolling_df['wa_PP_svPct' + str(weight)] = goalie_rolling_df['wa_powerPlaySaves' + str(weight)] / \
                                                         goalie_rolling_df['wa_powerPlayShotsAgainst' + str(weight)]
        goalie_rolling_df['wa_EV_svPct' + str(weight)] = goalie_rolling_df['wa_evenSaves' + str(weight)] / \
                                                         goalie_rolling_df['wa_evenShotsAgainst' + str(weight)]
        goalie_rolling_df['wa_svPct' + str(weight)] = goalie_rolling_df['wa_saves' + str(weight)] / goalie_rolling_df[
            'wa_shots' + str(weight)]

    return (goalie_rolling_df)


def goalie_xG_data(goalie_game_data,
                   game_info_data,
                   season_start,
                   goalie_rolling_df):
    result_feats = ['SA', 'Goal', 'xG_total']
    control_feats = ['hours_rest', 'travel_km']
    control_windows = [10]
    control_offsets = [24, 48]
    result_windows = [10, 20, 40]
    result_offsets = [48, 168]

    output_metrics = ['id', 'game_start_est', 'SA_Goalie_Id', 'SA', 'Goal', 'xG_total']

    last_id = max(goalie_rolling_df['id'])

    ## Prep data
    game_info_data['game_start_est'] = pd.to_datetime(game_info_data['game_start_est'], utc=True)

    game_info_update = game_info_data.loc[(game_info_data['status'] == 'Final') &
                                          (game_info_data.id.astype(float) > last_id), :]

    ## Find future games too
    game_info_future = game_info_data.loc[game_info_data['status'] != 'Final', :]
    game_info_future = expand_future_games_possible_starters(game_info_future, 2, False)

    game_info_update = game_info_update.append(game_info_future)

    game_info_update[['away_starter_id', 'home_starter_id']] = game_info_update[
        ['away_starter_id', 'home_starter_id']].astype(float).astype(int)

    print("New games to update for goalie data :" + str(game_info_update.shape))

    # Prior season metrics
    goalie_game = goalie_game_data.loc[round(goalie_game_data.season.astype(float) / 10000) >= (season_start - 1), :]

    game_info = game_info_data.loc[round(game_info_data.season.astype(float) / 10000) >= (season_start - 1), :]

    game_info_future2 = pd.melt(game_info_future.loc[:,
                                ['id', 'game_start_utc', 'game_start_est', 'city_lat', 'home_team', 'away_team',
                                 'city_long', 'away_starter_id', 'home_starter_id']],
                                id_vars=['id', 'game_start_utc', 'game_start_est', 'home_team', 'away_team', 'city_lat',
                                         'city_long'],
                                value_vars=['away_starter_id', 'home_starter_id'],
                                var_name='season',
                                value_name='SA_Goalie_Id') \
        .merge(goalie_game.loc[:, ['SA_Goalie', 'SA_Goalie_Id']].drop_duplicates(), on=['SA_Goalie_Id'], how='left') \
        .drop_duplicates()

    goalie_game = goalie_game \
        .merge(game_info.loc[:,
               ['id', 'game_start_est', 'home_starter_id', 'away_starter_id', 'away_team', 'home_team', 'city_lat',
                'city_long']] \
               , on='id', how='inner') \
        .sort_values(['id', 'game_start_est'], ascending=True)

    goalie_game = goalie_game.loc[(goalie_game['SA_Goalie_Id'] == goalie_game['home_starter_id']) |
                                  (goalie_game['SA_Goalie_Id'] == goalie_game['away_starter_id']), :]

    goalie_game = goalie_game \
        .append(game_info_future2) \
        .sort_values('id', ascending=True)

    goalie_game['season'] = goalie_game['id'].astype(str).str[:4]

    goalie_df = pd.DataFrame()

    goalie_id = list(game_info_update['home_starter_id'].append(game_info_update['away_starter_id']).drop_duplicates())

    print(str(len(goalie_id)) + " goalies updating")

    for goalie in goalie_id:
        print(goalie)
        goalie_select = goalie_game.loc[goalie_game.SA_Goalie_Id.astype(int) == goalie, :].reset_index(drop=True) \
            .sort_values('game_start_est', ascending=True)

        goalie_select_out = goalie_game.loc[goalie_game.SA_Goalie_Id.astype(int) == goalie, output_metrics].reset_index(
            drop=True) \
            .sort_values('game_start_est', ascending=True)

        if goalie_select_out.shape[0] > 0:

            ## Last game geolocation
            goalie_select['last_city_lat'] = goalie_select.groupby('season')['city_lat'].shift(1).fillna(
                method='bfill').fillna(method='ffill').fillna(np.median(game_info_data['city_lat']))
            goalie_select['last_city_long'] = goalie_select.groupby('season')['city_long'].shift(1).fillna(
                method='bfill').fillna(method='ffill').fillna(np.median(game_info_data['city_long']))

            ## Travel distance
            goalie_select['travel_km'] = goalie_select.apply(lambda x: vincenty((x.city_lat, x.city_long), \
                                                                                (
                                                                                    x.last_city_lat, x.last_city_long)),
                                                             axis=1) \
                                             .astype(str).str[:-3].astype(float)

            goalie_select['last_game_start_est'] = goalie_select['game_start_est'].shift(1)

            ## Hours from last game
            goalie_select['last_game_start_est'] = goalie_select['game_start_est'].shift(1)

            goalie_select['hours_rest'] = (pd.to_datetime(goalie_select.game_start_est, utc=True) - \
                                           pd.to_datetime(goalie_select.last_game_start_est, utc=True)). \
                astype('timedelta64[h]').fillna(24 * 14).astype(int)

            goalie_select['hours_rest'] = goalie_select['hours_rest'].apply(lambda x: (24 * 14) if x > (24 * 14) else x)
            goalie_select_out['hours_rest'] = goalie_select['hours_rest'].fillna(24 * 14)

            goalie_select_out['travel_km'] = goalie_select['travel_km'].fillna(1277)
            goalie_select_out['hours_rest'] = goalie_select['hours_rest'].fillna(24 * 14)
            goalie_select_out['days_rest'] = np.round(goalie_select_out['hours_rest'] / 24)

            # If too few games append replacement level data on
            if goalie_select_out.shape[0] < 50:
                ## Append replacement data to start
                goalie_select = replacement_xG_goalie_data(goalie_game_data, result_windows) \
                    .append(goalie_select, ignore_index=True)

                goalie_select_out = replacement_xG_goalie_data(goalie_game_data, result_windows)[
                    ['hours_rest', 'SA_Goalie_Id']] \
                    .append(goalie_select_out, ignore_index=True)

            goalie_select = goalie_select.fillna(0)

            for window in range(len(control_windows)):
                for hour_offset in control_offsets:
                    # For features known before game time
                    for i in control_feats:
                        output = roll(goalie_select, control_windows[window] + 1) \
                            .apply(
                            lambda x: weighted_mean_offset(x[i], x.hours_rest, control_windows[window],
                                                           hour_offset, 0)) \
                            .rename("wa_" + str(i) + "_w" + str(control_windows[window]) + "_o" + str(hour_offset))

                        goalie_select_out = goalie_select_out.join(output)

            for window in range(len(result_windows)):
                for hour_offset in result_offsets:
                    # For features as the result of the game
                    for i in result_feats:
                        output = roll(goalie_select, result_windows[window] + 1) \
                            .apply(lambda x: weighted_mean_offset(x[i], x.hours_rest, result_windows[window],
                                                                  hour_offset, 1)) \
                            .rename("wa_" + str(i) + "_w" + str(result_windows[window]) + "_o" + str(hour_offset))

                        goalie_select_out = goalie_select_out.join(output)

            goalie_select_out = goalie_select_out.loc[goalie_select_out.SA_Goalie_Id.astype(int) > 2, :]

        # Append
        goalie_df = goalie_df.append(goalie_select_out)

    ## Subset to new goalie data
    goalie_df = goalie_df.loc[goalie_df.id.astype(int) > last_id, :]
    goalie_rolling_df = goalie_rolling_df.append(goalie_df)

    weights = ['_w10_o48', '_w20_o48', '_w40_o48', '_w10_o168', '_w20_o168', '_w40_o168']

    for weight in weights:
        goalie_rolling_df['wa_GPAA100' + str(weight)] = (goalie_rolling_df['wa_xG_total' + str(weight)] - \
                                                         goalie_rolling_df['wa_Goal' + str(weight)]) / \
                                                        (goalie_rolling_df['wa_SA' + str(weight)] / 100)

    return (goalie_rolling_df)


## Create Function to Pull Roster and Game Data to Calculate % D/Miss Shots
def team_shooting(game_info_data, game_roster_data):
    """
    """
    team_shooting_info = read_boto_s3('games-all', 'team_shooting_info.csv')

    last_id = max(team_shooting_info['id'])

    game_info = game_info_data.loc[game_info_data['id'] > last_id, :]

    ## Miss rate
    game_info['home_SOG_rate'] = (
                game_info['home_shots'] / (game_info[['home_attempts', 'away_blocked', 'home_shots']].sum(axis=1)))
    game_info['away_SOG_rate'] = (
                game_info['away_shots'] / (game_info[['away_attempts', 'home_blocked', 'away_shots']].sum(axis=1)))

    ## Fwd Shot Share
    game_roster = game_roster_data.loc[game_roster_data['id'] > last_id, :]

    game_roster['fwd_shot'] = game_roster.apply(lambda x: x.shots if x.Pos != 'D' else 0, axis=1)

    game_shots = game_roster.loc[game_roster['Pos'] != 'G', :] \
        .groupby(['id', 'team'])['shots', 'fwd_shot'] \
        .agg({'shots': 'sum', 'fwd_shot': 'sum'}).reset_index()

    game_shots['fwd_shot_share'] = game_shots['fwd_shot'] / game_shots['shots']

    game_shots['team'] = game_shots['team'].astype(str) + '_F_shot_share'

    game_shots = game_shots \
        .pivot(index='id', columns='team', values='fwd_shot_share').reset_index()

    game_all = game_info.loc[:, ['id', 'home_SOG_rate', 'away_SOG_rate']] \
        .merge(game_shots, on='id', how='inner')

    team_shooting_info = team_shooting_info.append(game_all)

    return (team_shooting_info)


## Adjusted metrics game-level
# Time game time leading
def score_adjusted_game_pbp(scored_data):
    score_venue_multipliers = read_boto_s3('shots-all', 'score_venue_multipliers.csv')

    score_adjusted_game_data = read_boto_s3('games-all', 'score_adjusted_game_data.csv')
    last_id = max(score_adjusted_game_data['id'])

    data = scored_data.loc[:,
           ['season', 'Game_Id', 'Home_Score', 'Away_Score', 'xG_raw', 'is_Rebound', 'is_Rush', 'Event', 'Home_Shooter',
            'Period',
            'Seconds_Elapsed', 'Home_Players', 'Away_Players']]

    data['id'] = ((round(data['season'] / 10000) * 1000000) + data['Game_Id']).astype(int)

    data = data.loc[data['id'] > last_id, :]

    data['xG_lag'] = data.groupby('Home_Shooter')['xG_raw'].shift()

    data['xG_team'] = data.apply(lambda x: x['xG_raw'] if x['is_Rebound'] == 0
    else x['xG_raw'] * (1 - x['xG_lag'])
                                 , axis=1)

    data['home_score_state'] = data['Home_Score'] - data['Away_Score']
    data = data.dropna()

    data['home_score_state'] = data \
        .apply(lambda x: 3 if x.home_score_state > 2
    else -3 if x.home_score_state < -2
    else int(x.home_score_state), axis=1)

    data['home_score_state2'] = 'Duration' + data['home_score_state'].astype(str)

    data['home_attempts'] = data['Home_Shooter']
    data['away_attempts'] = 1 - data['Home_Shooter']

    data['even_strength'] = data.apply(lambda x: 1 if (x.Home_Players - x.Away_Players) == 0 else 0, axis=1)
    data['home_advantage'] = data.apply(lambda x: 1 if (x.Home_Players - x.Away_Players) > 0 else 0, axis=1)
    data['away_advantage'] = data.apply(lambda x: 1 if (x.Home_Players - x.Away_Players) < 0 else 0, axis=1)

    data['game_seconds'] = ((data['Period'] - 1) * (20 * 60)) + data['Seconds_Elapsed']

    goals = data.loc[(data['Event'] == 'GOAL'), :]

    buzzer = goals.groupby('id')['game_seconds'].max().reset_index()

    buzzer['game_seconds'] = buzzer['game_seconds'].apply(lambda x: x if x > 3600 else 3600)

    goals = goals.merge(buzzer, on=['id', 'game_seconds'], how='outer')

    goals = goals.sort_values(['id', 'game_seconds']).fillna(method='ffill')

    goals['duration_seconds'] = goals['game_seconds'] - goals.groupby(['id'])['game_seconds'].shift().fillna(0)

    game_score_state = goals.groupby(['id', 'home_score_state2'])['duration_seconds'].sum().reset_index() \
        .pivot(index='id', columns='home_score_state2')['duration_seconds'].reset_index().fillna(0)
    print(game_score_state.columns)

    data = data \
        .merge(score_venue_multipliers, on='home_score_state', how='left')

    data['home_SF_adj'] = data['home_attempts'] * data['home_coef']
    data['away_SF_adj'] = data['away_attempts'] * data['away_coef']

    data['home_xGF_adj'] = data['home_attempts'] * data['xG_team'] * data['home_xG_coef']
    data['away_xGF_adj'] = data['away_attempts'] * data['xG_team'] * data['away_xG_coef']

    ## xG Strength
    data['home_EV_xGF_adj'] = data['home_attempts'] * data['xG_team'] * data['home_xG_coef'] * data['even_strength']
    data['away_EV_xGF_adj'] = data['away_attempts'] * data['xG_team'] * data['away_xG_coef'] * data['even_strength']

    data['home_PP_xGF_adj'] = data['home_attempts'] * data['xG_team'] * data['home_xG_coef'] * data['home_advantage']
    data['away_PP_xGF_adj'] = data['away_attempts'] * data['xG_team'] * data['away_xG_coef'] * data['away_advantage']

    data['home_PK_xGF_adj'] = data['home_attempts'] * data['xG_team'] * data['home_xG_coef'] * data['away_advantage']
    data['away_PK_xGF_adj'] = data['away_attempts'] * data['xG_team'] * data['away_xG_coef'] * data['home_advantage']

    data['home_rush_attempt'] = data['home_attempts'] * data['is_Rush']
    data['away_rush_attempt'] = data['away_attempts'] * data['is_Rush']

    data['home_rebound_attempt'] = data['home_attempts'] * data['is_Rebound']
    data['away_rebound_attempt'] = data['away_attempts'] * data['is_Rebound']

    game_fenwick_totals = data.groupby(['id', 'season'])[
        ['home_SF_adj', 'away_SF_adj', 'home_xGF_adj', 'away_xGF_adj',
         'home_EV_xGF_adj', 'away_EV_xGF_adj', 'home_PP_xGF_adj', 'away_PP_xGF_adj', 'home_PK_xGF_adj',
         'away_PK_xGF_adj',
         'home_rush_attempt', 'away_rush_attempt', 'home_rebound_attempt', 'away_rebound_attempt']].sum().reset_index()

    print(game_fenwick_totals.columns)

    game_fenwick_totals = game_fenwick_totals.merge(game_score_state, on='id', how='left')

    score_adjusted_game_data = score_adjusted_game_data.append(game_fenwick_totals)

    duration_metrics = ['Duration-1', 'Duration-2', 'Duration-3', 'Duration0', 'Duration1', 'Duration2', 'Duration3']
    score_adjusted_game_data['Duration_60'] = score_adjusted_game_data[duration_metrics].sum(axis=1).fillna(3600)
    score_adjusted_game_data['Duration_60'] = score_adjusted_game_data['Duration_60'].apply(
        lambda x: x if x >= 3600 else 3600)

    return (score_adjusted_game_data)

def goalie_game_update(start_game_id, scored_data):
    # Prior games
    goalie_game_data = read_boto_s3('games-all', "goalie_game_data.csv")

    rebound_goal_probability = 0.27
    goalie_shot_data = scored_data.copy()

    goalie_shot_data['id'] = (
                (round(goalie_shot_data['season'] / 10000) * 1000000) + goalie_shot_data['Game_Id']).astype(int)

    goalie_shot_data = goalie_shot_data.loc[goalie_shot_data['id'] > start_game_id, :]

    goalie_shot_data['SA'] = 1
    goalie_shot_data['xG_RB'] = goalie_shot_data['xR'].apply(lambda x: x * rebound_goal_probability)
    goalie_shot_data['xG_NRB'] = goalie_shot_data.apply(lambda x: x.xG_raw if x.is_Rebound == 0 else 0, axis=1)
    goalie_shot_data['xG_total'] = goalie_shot_data['xG_NRB'] + goalie_shot_data['xG_RB']

    goalie_game_data_new = goalie_shot_data \
        .groupby(['season', 'id', 'Game_Id', 'SA_Goalie', 'SA_Goalie_Id']) \
        ['SA', 'Goal', 'xG_total', 'xG_NRB', 'xG_RB', 'xG_raw', 'is_Rebound', 'xR'] \
        .sum() \
        .reset_index()

    goalie_game_data = goalie_game_data.append(goalie_game_data_new)

    return (goalie_game_data)


def goalie_xG_game_features(game_info_data,
                         goalie_rolling_df,
                         control_feats = ['hours_rest'],
                         result_weights=['_w10_o48', '_w20_o48', '_w40_o48', '_w10_o168', '_w20_o168', '_w40_o168'],
                         control_weights=['_w10_o24', '_w10_o48']):
    """
    Converts goalie-level data into game-starter level data, assigning to home and away files
    :param game_info_data: all games
    :param goalie_rolling_df: goalie-level data
    :param goalie_features: goalie-features to keep
    :param weights:
    :return:
    """

    goalie_rolling_df['id'] = goalie_rolling_df['id'].astype(int)
    goalie_rolling_df['SA_Goalie_Id'] = goalie_rolling_df['SA_Goalie_Id'].astype(int)

    game_info_data['id'] = game_info_data['id'].astype(int)
    game_info_data['home_starter_id'] = game_info_data['home_starter_id'].astype(int)
    game_info_data['away_starter_id'] = game_info_data['away_starter_id'].astype(int)


    result_features = [str(feat) + str(w) for feat in ['wa_GPAA100'] for w in
                       result_weights]
    control_features = [str(feat) + str(w) for feat in ['wa_hours_rest', 'wa_travel_km'] for w in control_weights]

    ## Find 80/20 values to impute
    p50_results = goalie_rolling_df.loc[:,
                  [str(feat) + str(w) for feat in ['wa_hours_rest', 'wa_travel_km'] for w in control_weights]
                        + control_feats] \
        .dropna(axis=0, how='any') \
        .quantile(q=0.5, axis=0)

    p10_results = goalie_rolling_df.loc[:,
                  [str(feat) + str(w) for feat in ['wa_GPAA100'] for w in result_weights]] \
        .dropna(axis=0, how='any') \
        .quantile(q=0.1, axis=0)

    ## Home start metrics
    home_goalie_metrics = game_info_data \
                              .loc[:, ['id', 'home_starter_id']] \
        .drop_duplicates() \
        .merge(goalie_rolling_df.loc[:, ['id', 'SA_Goalie_Id'] + control_features + result_features + control_feats],
               left_on=['id', 'home_starter_id'],
               right_on=['id', 'SA_Goalie_Id'],
               how='left') \
        .fillna(value=p50_results.append(p10_results)) \
        .drop(['SA_Goalie_Id'], axis=1)

    home_goalie_metrics.columns = ['id', 'home_starter_id'] + ['home_starter_' + str(feat) for feat in
                                                               control_features + result_features + control_feats]

    away_goalie_metrics = game_info_data \
                              .loc[:, ['id', 'away_starter_id']] \
        .drop_duplicates() \
        .merge(goalie_rolling_df.loc[:, ['id', 'SA_Goalie_Id'] + control_features + result_features + control_feats],
               left_on=['id', 'away_starter_id'],
               right_on=['id', 'SA_Goalie_Id'],
               how='left') \
        .fillna(value=p50_results.append(p10_results)) \
        .drop(['SA_Goalie_Id'], axis=1)

    away_goalie_metrics.columns = ['id', 'away_starter_id'] + ['away_starter_' + str(feat) for feat in
                                                               control_features + result_features + control_feats]

    return (home_goalie_metrics, away_goalie_metrics)


def df_feature_transformer(model_df):

    model_df = model_df.copy()

    result_weights = ['_w10_o48', '_w20_o48', '_w40_o48', '_w10_o168', '_w20_o168', '_w40_o168']

    ## PK Efficiency
    pkga_vars = [str(venue) + str(feat) + str(w) for feat in ['_wa_game_PKGA']
                 for venue in ['home','away']
                 for w in result_weights]
    pkatt_vars = [str(venue) + str(feat) + str(w) for feat in ['_wa_game_PKAtt']
                 for venue in ['home','away']
                 for w in result_weights]

    ## PP Efficiency
    ppga_vars = [str(venue) + str(feat) + str(w) for feat in ['_wa_game_PPGF']
                 for venue in ['home','away']
                 for w in result_weights]
    ppatt_vars = [str(venue) + str(feat) + str(w) for feat in ['_wa_game_PPAtt']
                 for venue in ['home','away']
                 for w in result_weights]

    for i in range(len(pkga_vars)):
        model_df[pkga_vars[i]] = model_df[pkga_vars[i]] / model_df[pkatt_vars[i]]
        model_df[ppga_vars[i]] = model_df[ppga_vars[i]] / model_df[ppatt_vars[i]]

    model_df.columns = model_df.columns.str.replace('PKGA','PKEff')\
                                       .str.replace('PPGF','PPEff')


    ## Variables to remove
    travel_vars = ['travel_km', 'wa_travel_km_w10_o48','starter_wa_travel_km_w10_o24','starter_wa_travel_km_w10_o48']
    time_vars = ['starter_wa_hours_rest_w10_o48'] # Only need one

    model_df['away_ln_km_perday'] = np.log(0.01 + (model_df['away_travel_km'] / (model_df['away_hours_rest']/24)))
    model_df['home_ln_km_perday'] = np.log(0.01 + (model_df['home_travel_km'] / (model_df['home_hours_rest']/24)))
    model_df['away_travel_index'] = model_df['away_wa_travel_km_w10_o48'] / (model_df['away_wa_hours_rest_w10_o48']/24)
    model_df['home_travel_index'] = model_df['home_wa_travel_km_w10_o48'] / (model_df['home_wa_hours_rest_w10_o48']/24)

    model_df = model_df.drop([str(venue) + str(feat)
                 for venue in ['home_','away_']
                 for feat in travel_vars + time_vars], axis=1)

    return(model_df)