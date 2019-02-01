#!/usr/bin/python
"""
This module contains functions that scrapes games and assembles a game-level modeling dataframe and scores it
"""
import scrape_functions
import pipeline_functions
import datetime as dt

end_game_id = pipeline_functions.yesterday_last_gameid()
year = 2018

print("Update until game id: " + str(end_game_id))

# Only re-run model on Mondays
if dt.date.today().isoweekday() == 1:
    print("Modeling and Scoring Data")
    scrape_functions.scrape_games(end_game_id, None, year, True, True, True)
else:
    print("Scoring Data")
    scrape_functions.scrape_games(end_game_id, None, year, True, True, False)
