#!/usr/bin/python
"""
This module contains functions that scrapes games and assembles a game-level modeling dataframe and scores it
"""
import time
print("Running game_pipeline.py")

# Sleep for 4 hours
#time.sleep(6.5*60*60)

print("Waking up")# + str(datetime.now()))

import scrape_functions
import pipeline_functions
import model_functions
import xG_update


# ## Read data from s3
arena_geocode = pipeline_functions.read_boto_s3('games-all','arena_geocode.csv')
game_info_data = pipeline_functions.read_boto_s3('games-all','game_info_data.csv')
game_roster_data = pipeline_functions.read_boto_s3('games-all','game_roster_data.csv')
team_day_ratings_lag = pipeline_functions.read_boto_s3('games-all','team_day_ratings_lag.csv')
team_day_elos = pipeline_functions.read_boto_s3('games-all','team_day_elos.csv')
team_game_features = pipeline_functions.read_boto_s3('games-all','team_game_features.csv')
game_level_model_df = pipeline_functions.read_boto_s3('games-all','game_level_model_df.csv')
goalie_rolling_df = pipeline_functions.read_boto_s3('games-all','goalie_rolling_df.csv')
goalie_xG_rolling_df = pipeline_functions.read_boto_s3('games-all','goalie_xG_rolling_df.csv')
scored_data2017_2019 = pipeline_functions.read_boto_s3('shots-all', "scored_data2017_2019.csv")

start_game_id = max(game_info_data['id'])
end_game_id = pipeline_functions.today_last_gameid()
year = 2018

goalie_IR = []

print(start_game_id)

print("Scrape from " + str(start_game_id+1) + " to " + str(year) + "0" + str(end_game_id))

if int(str(year) + "0" + str(end_game_id)) > start_game_id:

    print("Processing " + str(int(str(year) + "0" + str(end_game_id)) - start_game_id) + " games")
    print("Game-level dataset: " + str(game_info_data.shape))

    game_info_data2, game_roster_data2, team_day_elos2, team_day_ratings_lag2 = pipeline_functions.process_games_ytd(year,
                                                     end_game_id,
                                                     game_info_data,
                                                     game_roster_data,
                                                     team_day_ratings_lag,
                                                     team_day_elos,
                                                     arena_geocode)

    print(game_info_data2.shape)

    pipeline_functions.write_boto_s3(game_info_data2, 'games-all', 'game_info_data2.csv')
    pipeline_functions.write_boto_s3(game_roster_data2, 'games-all', 'game_roster_data2.csv')
    pipeline_functions.write_boto_s3(team_day_ratings_lag2, 'games-all', 'team_day_ratings_lag2.csv')
    pipeline_functions.write_boto_s3(team_day_elos2, 'games-all', 'team_day_elos2.csv')


    ## Goalie shot-level data
    goalie_rolling_df2 = pipeline_functions.goalie_data(game_roster_data2,
                                    game_info_data2,
                                    2015,
                                    goalie_rolling_df)
    pipeline_functions.write_boto_s3(goalie_rolling_df2, 'games-all', 'goalie_rolling_df2.csv')


    game_info_data2 = pipeline_functions.read_boto_s3('games-all', 'game_info_data2.csv')
    game_roster_data2 = pipeline_functions.read_boto_s3('games-all', 'game_roster_data2.csv')
    team_day_elos2 = pipeline_functions.read_boto_s3('games-all', 'team_day_elos2.csv')
    team_day_ratings_lag2 = pipeline_functions.read_boto_s3('games-all', 'team_day_ratings_lag2.csv')
    game_level_model_df = pipeline_functions.read_boto_s3('games-all', 'game_level_model_df.csv')

    ## Update goalie game data
    goalie_game_data2 = pipeline_functions.goalie_game_update(start_game_id, scored_data2017_2019)
    pipeline_functions.write_boto_s3(goalie_game_data2, 'games-all', 'goalie_game_data2.csv')

    ## Goalie data
    goalie_xG_rolling_df2 = pipeline_functions.goalie_xG_data(goalie_game_data2,
                   game_info_data2,
                   year,
                   goalie_xG_rolling_df)

    pipeline_functions.write_boto_s3(goalie_xG_rolling_df2, 'games-all', 'goalie_xG_rolling_df2.csv')
    goalie_xG_rolling_df2 = pipeline_functions.read_boto_s3('games-all', 'goalie_xG_rolling_df2.csv')

    ## Shooter data
    team_shooting_info2 = pipeline_functions.team_shooting(game_info_data2, game_roster_data2)
    pipeline_functions.write_boto_s3(team_shooting_info2, 'games-all', 'team_shooting_info2.csv')

    ## Team Adjusted data
    score_adjusted_game_data2 = pipeline_functions.score_adjusted_game_pbp(scored_data2017_2019)
    pipeline_functions.write_boto_s3(score_adjusted_game_data2, 'games-all', 'score_adjusted_game_data2.csv')

    pipeline_functions.write_boto_s3(pipeline_functions.possible_starters(game_info_data2, game_roster_data2), 'games-all', 'possible-starters.csv')

    model_df2, team_game_features2 = pipeline_functions.team_game_features_ytd(year,
                                                    game_info_data2,
                                                    team_day_elos2,
                                                    score_adjusted_game_data2,
                                                    team_shooting_info2,
                                                    team_game_features,
                                                    goalie_xG_rolling_df2,
                                                    game_level_model_df,
                                                    goalie_IR)

    pipeline_functions.write_boto_s3(model_df2, 'games-all', 'game_level_model_df2.csv')
    pipeline_functions.write_boto_s3(team_game_features2, 'games-all', 'team_game_features2.csv')

    model_df2 = pipeline_functions.read_boto_s3('games-all', 'game_level_model_df2.csv')

    scored_game_probs = model_functions.score_game_data(model_df2,
                                        model_list=['lr_cvsearch', #'gnb_isotonic', 'rf_isotonic',# 'mlp_isotonic',
                                                    'lr_model'],  # xgb_model need to re-run
                                        control_windows=[10],
                                        control_offsets=[48],
                                        result_windows=[20, 40],
                                        result_offsets=[48, 168])

    ensembled_game_probs = model_functions.ensemble_models(scored_game_probs,
                                           model_list=['lr_cvsearch', 'lr_model'],
                                           # 'gnb_isotonic', '', 'mlp_isotonic', #xgb_model
                                           result_windows=[10, 20, 40],
                                           result_offsets=[48, 168],
                                           elo_metrics=['elo_k4_wm4_SO2'],
                                           custom_model_list=['lr_cvsearch_custom',
                                                              # 'rf_isotonic_custom', 'gnb_isotonic_custom', mlp_isotonic_custom
                                                              'lr_model_custom', 'svm_model_custom'])

    pipeline_functions.write_boto_s3(scored_game_probs, 'games-all', 'scored_game_probs.csv')
    pipeline_functions.write_boto_s3(ensembled_game_probs, 'games-all', 'ensembled_game_probs.csv')

    goalie_prediction_matrix = model_functions.output_goalie_prediction_probs(ensembled_game_probs)
    model_functions.output_goalie_matrix(goalie_prediction_matrix)

else:
    print("Update game ID")