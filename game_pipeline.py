"""
This module contains functions that scrapes games and assembles a game-level modeling dataframe and scores it
"""
import scrape_functions
import pipeline_functions

## Read data from s3
game_info_data = pipeline_functions.read_boto_s3('games-all','game_info_data.csv')
# game_roster_data = pipeline_functions.read_boto_s3('games-all','game_roster_data.csv')
# team_day_ratings_lag = pipeline_functions.read_boto_s3('games-all','team_day_ratings_lag.csv')
# arena_geocode = pipeline_functions.read_boto_s3('games-all','arena_geocode.csv')
# team_game_features = pipeline_functions.read_boto_s3('games-all','team_game_features.csv')
# game_level_model_df = pipeline_functions.read_boto_s3('games-all','game_level_model_df.csv')
# goalie_rolling_df = pipeline_functions.read_boto_s3('games-all','goalie_rolling_df.csv')

start_game_id = max(game_info_data['id'])
end_game_id = pipeline_functions.yesterday_last_gameid()
year = 2018

print(start_game_id)

## Update xG model
scrape_functions.scrape_games(end_game_id, None, year, True, False, True)


print("Scrape from " + str(start_game_id+1) + " to " + str(year) + "0" + str(end_game_id))

if int(str(year) + "0" + str(end_game_id)) > start_game_id:

    print("Processing " + str(int(str(year) + "0" + str(end_game_id)) - start_game_id) + " games")
    print("Game-level dataset: " + str(game_info_data.shape))

    # game_info_data2, game_roster_data2, team_day_elos2, team_day_ratings_lag2 = pipeline_functions.process_games_ytd(year,
    #                                                  end_game_id,
    #                                                  game_info_data,
    #                                                  game_roster_data,
    #                                                  team_day_ratings_lag,
    #                                                  arena_geocode)
    #
    # print(game_info_data2.shape)
    #
    # goalie_rolling_df2 = pipeline_functions.goalie_data(game_roster_data2,
    #                                                 game_info_data2,
    #                                                 2015,
    #                                                 goalie_rolling_df,
    #                                                 windows = [6,12],
    #                                                 hour_offsets = [24, 48])
    #
    #
    #
    # pipeline_functions.write_boto_s3(game_info_data2, 'games-all', 'game_info_data2.csv')
    # pipeline_functions.write_boto_s3(game_roster_data2, 'games-all', 'game_roster_data2.csv')
    # pipeline_functions.write_boto_s3(team_day_ratings_lag2, 'games-all', 'team_day_ratings_lag2.csv')
    # pipeline_functions.write_boto_s3(goalie_rolling_df2, 'games-all', 'goalie_rolling_df2.csv')
    # pipeline_functions.write_boto_s3(team_day_elos2, 'games-all', 'team_day_elos2.csv')
    #
    #
    # model_df2, team_game_features2 = pipeline_functions.team_game_features_ytd(year,
    #                                                 game_info_data2,
    #                                                 team_day_elos2,
    #                                                 team_game_features,
    #                                                 goalie_rolling_df2,
    #                                                 game_level_model_df)
    #
    # pipeline_functions.write_boto_s3(model_df2, 'games-all', 'game_level_model_df2.csv')
    # pipeline_functions.write_boto_s3(team_game_features2, 'games-all', 'team_game_features2.csv')

    # model_df2 = pipeline_functions.read_boto_s3('games-all','game_level_model_df2.csv')
    #
    #
    # all_game_probs = pipeline_functions.model_df(model_df2,
    #               windows = [8, 12],
    #               hour_offsets = [24, 48],
    #               elo_metrics = ['elo_k4_wm4', 'elo_k4_wm4_SO2', 'elo_k4_wm2_SO2'],
    #               model_list = ['lr_cvsearch', 'gnb_isotonic', 'rf_isotonic', 'mlp_isotonic', 'lr_model'],
    #               game_features=['game_GF', 'game_GA', 'game_SF', 'game_SA', 'game_PPGF','game_PKGA', 'game_PPAtt','game_PKAtt'],
    #               standard_features = ['hours_rest', 'travel_km'])
    #
    # pipeline_functions.write_boto_s3(all_game_probs, 'games-all', 'all_game_probs.csv')

else:
    print("Update game ID")