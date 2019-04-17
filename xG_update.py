#!/usr/bin/python
"""
This module contains functions that scrapes games and assembles a game-level modeling dataframe and scores it
"""
import scrape_functions
import pipeline_functions
import numpy as np
from datetime import date, timedelta

end_game_id = pipeline_functions.yesterday_last_gameid()
tomorrow_date = date.today() + timedelta(1)
year = 2018

if date.today() > date(2019, 4, 10):

    print("Update playoff until date: " + str(tomorrow_date))
    print("Scoring Data")
    scrape_functions.scrape_games(end_game_id, tomorrow_date, year, True, True, False)

else:

    print("Update regular season until game id: " + str(end_game_id))

    # Only re-run model on Mondays
    if date.today().isoweekday() == 1:
        print("Modeling and Scoring Data")
        scrape_functions.scrape_games(end_game_id, None, year, True, True, True)
    else:
        print("Scoring Data")
        scrape_functions.scrape_games(end_game_id, None, year, True, True, False)
