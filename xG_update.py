"""
This module contains functions that scrapes games and assembles a game-level modeling dataframe and scores it
"""
import scrape_functions
import pipeline_functions
end_game_id = pipeline_functions.yesterday_last_gameid()
year = 2018

print("Update until game id: " + str(end_game_id))

## Update xG model
scrape_functions.scrape_games(end_game_id, None, year, True, False, True)

