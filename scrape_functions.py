import numpy as np
import json_pbp
import html_pbp
import espn_pbp
from sklearn.preprocessing import PolynomialFeatures
import json_shifts
import json_schedule
import html_shifts
import playing_roster
import json_schedule
import pandas as pd
import time
import datetime
import shared
import thinkbayes2 as tb
import df_adjustments
import pipeline_functions

##########
# Holds list for broken games for shifts and pbp
broken_shifts_games = []
broken_pbp_games = []
players_missing_ids = []
espn_games = []

columns = ['Game_Id', 'Date', 'Period', 'Event', 'Description', 'Time_Elapsed', 'Seconds_Elapsed', 'Strength',
           'Ev_Zone', 'Type', 'Ev_Team', 'Home_Zone', 'Away_Team', 'Home_Team', 'p1_name', 'p1_ID', 'p2_name', 'p2_ID',
           'p3_name', 'p3_ID', 'awayPlayer1', 'awayPlayer1_id', 'awayPlayer2', 'awayPlayer2_id', 'awayPlayer3',
           'awayPlayer3_id', 'awayPlayer4', 'awayPlayer4_id', 'awayPlayer5', 'awayPlayer5_id', 'awayPlayer6',
           'awayPlayer6_id', 'homePlayer1', 'homePlayer1_id', 'homePlayer2', 'homePlayer2_id', 'homePlayer3',
           'homePlayer3_id', 'homePlayer4', 'homePlayer4_id', 'homePlayer5', 'homePlayer5_id', 'homePlayer6',
           'homePlayer6_id', 'Away_Players', 'Home_Players', 'Away_Score', 'Home_Score', 'Away_Goalie',
           'Away_Goalie_Id', 'Home_Goalie', 'Home_Goalie_Id', 'xC', 'yC', 'Home_Coach', 'Away_Coach']


def check_goalie(row):
    """
    Checks for bad goalie names (you can tell by them having no player id)
    :param row: df row
    """
    if row['Away_Goalie'] != '' and row['Away_Goalie_Id'] == 'NA':
        players_missing_ids.extend([row['Away_Goalie'], row['Game_Id']])

    if row['Home_Goalie'] != '' and row['Home_Goalie_Id'] == 'NA':
        players_missing_ids.extend([row['Home_Goalie'], row['Game_Id']])


def get_players_json(json):
    """
    Return dict of players for that game
    :param json: gameData section of json
    :return: dict of players->keys are the name (in uppercase)
    """
    players = dict()

    players_json = json['players']
    for key in players_json.keys():
        name = shared.fix_name(players_json[key]['fullName'].upper())
        players[name] = {'id': ' '}
        try:
            players[name]['id'] = players_json[key]['id']
        except KeyError:
            print(name, ' is missing an ID number')
            players[name]['id'] = 'NA'

    return players


def combine_players_lists(json_players, roster_players, game_id):
    """
    Combine the json list of players (which contains id's) with the list in the roster html
    :param json_players: dict of all players with id's
    :param roster_players: dict with home and and away keys for players
    :param game_id: id of game
    :return: dict containing home and away keys -> which contains list of info on each player
    """
    home_players = dict()
    for player in roster_players['Home']:
        try:
            name = shared.fix_name(player[2])
            id = json_players[name]['id']
            home_players[name] = {'id': id, 'number': player[0]}
        except KeyError:
            # This usually means it's the backup goalie (who didn't play) so it's no big deal with them
            if player[1] != 'G':
                players_missing_ids.extend([player, game_id])
                home_players[name] = {'id': 'NA', 'number': player[0]}

    away_players = dict()
    for player in roster_players['Away']:
        try:
            name = shared.fix_name(player[2])
            id = json_players[name]['id']
            away_players[name] = {'id': id, 'number': player[0]}
        except KeyError:
            if player[1] != 'G':
                players_missing_ids.extend([player, game_id])
                away_players[name] = {'id': 'NA', 'number': player[0]}

    return {'Home': home_players, 'Away': away_players}


def combine_html_json_pbp(json_df, html_df, game_id, date):
    """
    Join both data sources
    :param json_df: json pbp DataFrame
    :param html_df: html pbp DataFrame
    :param game_id: id of game
    :param date: date of game
    :return: finished pbp

    Add game_id and date
    Get rid of period, event, time_elapsed
    """
    try:
        html_df.Period = html_df.Period.astype(int)
        game_df = pd.merge(html_df, json_df, left_on=['Period', 'Event', 'Seconds_Elapsed'],
                           right_on=['period', 'event', 'seconds_elapsed'], how='left')

        # This id because merge doesn't work well with shootouts
        game_df = game_df.drop_duplicates(subset=['Period', 'Event', 'Description', 'Seconds_Elapsed'])
        game_df['Game_Id'] = game_id[-5:]
        game_df['Date'] = date
        return pd.DataFrame(game_df, columns=columns)
    except Exception as e:
        print('Problem combining Html Json pbp for game {}'.format(game_id, e))


def combine_espn_html_pbp(html_df, espn_df, game_id, date, away_team, home_team):
    """
    Merge the coordinate from the espn feed into the html DataFrame
    :param html_df: dataframe with info from html pbp
    :param espn_df: dataframe with info from espn pbp
    :param game_id: json game id
    :param date: ex: 2016-10-24
    :param away_team: away team
    :param home_team: home team
    :return: merged DataFrame
    """
    try:
        espn_df.period = espn_df.period.astype(int)
        df = pd.merge(html_df, espn_df, left_on=['Period', 'Seconds_Elapsed', 'Event'],
                      right_on=['period', 'time_elapsed', 'event'], how='left')

        # df = df.drop_duplicates(subset=['Period', 'Event', 'Seconds_Elapsed'])
        df = df.drop(['period', 'time_elapsed', 'event'], axis=1)
    except Exception as e:
        print('Error for combining espn and html pbp for game {}'.format(game_id), e)
        return None

    df['Game_Id'] = game_id[-5:]
    df['Date'] = date
    df['Away_Team'] = away_team
    df['Home_Team'] = home_team

    return pd.DataFrame(df, columns=columns)


def scrape_pbp(game_id, date, roster):
    """
    Scrapes the pbp
    Automatically scrapes the json and html, if the json is empty the html picks up some of the slack and the espn
    xml is also scraped for coordinates
    :param game_id: json game id
    :param date: date of game
    :param roster: list of players in pre game roster
    :return: DataFrame with info or None if it fails
             a dict of players with id's and numbers
    """
    game_json = json_pbp.get_pbp(game_id)
    try:
        teams = json_pbp.get_teams(game_json)  # Get teams from json
        player_ids = get_players_json(game_json['gameData'])
        players = combine_players_lists(player_ids, roster['players'], game_id)  # Combine roster names with player id's
    except Exception as e:
        print('Problem with getting the teams or players', e)
        return None, None

    year = str(game_id)[:4]
    # Coordinates are only available in json from 2010 onwards
    if int(year) >= 2010:
        try:
            json_df = json_pbp.parse_json(game_json)
            num_json_plays = len(game_json['liveData']['plays']['allPlays'])
        except Exception as e:
            print('Error for Json pbp for game {}'.format(game_id), e)
            return None, None
    else:
        num_json_plays = 0

    # Check if the json is missing the plays...if it is enable the HTML parsing to do more work to make up for the
    # json and scrape ESPN for the coordinates
    if num_json_plays == 0:
        espn_games.extend([game_id])
        html_df = html_pbp.scrape_game(game_id, players, teams, False)
        espn_df = espn_pbp.scrape_game(date, teams['Home'], teams['Away'])
        game_df = combine_espn_html_pbp(html_df, espn_df, str(game_id), date, teams['Away'], teams['Home'])
    else:
        html_df = html_pbp.scrape_game(game_id, players, teams, True)
        game_df = combine_html_json_pbp(json_df, html_df, str(game_id), date)

    if game_df is not None:
        game_df['Home_Coach'] = roster['head_coaches']['Home']
        game_df['Away_Coach'] = roster['head_coaches']['Away']

    return game_df, players


def scrape_shifts(game_id, players):
    """
    Scrape the Shift charts (or TOI tables)
    :param game_id: json game id
    :param players: dict of players with numbers and id's
    :return: DataFrame with info or None if it fails
    """
    year = str(game_id)[:4]
    try:
        if int(year) < 2010:  # Control for fact that shift json is only available from 2010 onwards
            raise Exception
        shifts_df = json_shifts.scrape_game(game_id)
    except Exception:
        try:
            shifts_df = html_shifts.scrape_game(game_id, players)
        except Exception as e:
            broken_shifts_games.extend([game_id])
            print('Error for html shifts for game {}'.format(game_id), e)
            return None

    return shifts_df


def scrape_game(game_id, date, if_scrape_shifts):
    """
    This scrapes the info for the game.
    The pbp is automatically scraped, and the whether or not to scrape the shifts is left up to the user
    :param game_id: game to scrap
    :param date: ex: 2016-10-24
    :param if_scrape_shifts: boolean, check if scrape shifts
    :return: DataFrame of pbp info
             (optional) DataFrame with shift info
    """

    print(' '.join(['Scraping Game ', game_id, date]))
    shifts_df = None

    try:
        roster = playing_roster.scrape_roster(game_id)
    except Exception:
        broken_pbp_games.extend([game_id, date])
        return None, None  # Everything fails without the roster

    pbp_df, players = scrape_pbp(game_id, date, roster)

    if pbp_df is None:
        broken_pbp_games.extend([game_id, date])

    if if_scrape_shifts and pbp_df is not None:
        shifts_df = scrape_shifts(game_id, players)

    return pbp_df, shifts_df


def scrape_list_of_games(games, if_scrape_shifts):
    """
    Given a list of game_id's (and a date for each game) it scrapes them
    :param games: list of [game_id, date]
    :param if_scrape_shifts: whether to scrape shifts
    :return: DataFrame of pbp info, also shifts if specified
    """
    pbp_dfs = []
    shifts_dfs = []

    for game in games:
        pbp_df, shifts_df = scrape_game(str(game[0]), game[1], if_scrape_shifts)
        if pbp_df is not None:
            pbp_dfs.extend([pbp_df])
        if shifts_df is not None:
            shifts_dfs.extend([shifts_df])

    # Check if any games
    if len(pbp_dfs) == 0:
        return None, None

    pbp_df = pd.concat(pbp_dfs)
    pbp_df = pbp_df.reset_index(drop=True)
    pbp_df.apply(lambda row: check_goalie(row), axis=1)

    if if_scrape_shifts:
        shifts_df = pd.concat(shifts_dfs)
        shifts_df = shifts_df.reset_index(drop=True)
    else:
        shifts_df = None

    return pbp_df, shifts_df


def scrape_date_range(from_date, to_date, if_scrape_shifts):
    """
    Scrape games in given date range
    :param from_date: date you want to scrape from
    :param to_date: date you want to scrape to
    :param if_scrape_shifts: boolean, check if scrape shifts
    """
    from sqlalchemy import create_engine

    engine = create_engine(
        'mysql+mysqlconnector://cole92anderson:cprice31!@css-db.cnqvzrgc2pnj.us-east-1.rds.amazonaws.com:3306/nhl_all')

    try:
        if time.strptime(to_date, "%Y-%m-%d") < time.strptime(from_date, "%Y-%m-%d"):
            print("Error: The second date input is earlier than the first one")
            return
    except ValueError:
        print("Incorrect format given for dates. They must be given like '2016-10-01' ")
        return

    games = json_schedule.scrape_schedule(from_date, to_date)
    pbp_df, shifts_df = scrape_list_of_games(games, if_scrape_shifts)

    if pbp_df is not None:
        pbp_df.to_csv('nhl_pbp{}{}.csv'.format(from_date, to_date), sep=',', index=False)
        # pbp_df.to_sql('nhl_pbp20172018',con=engine, if_exists='append', index=False)
    if shifts_df is not None:
        shifts_df.to_csv('nhl_shifts{}{}.csv'.format(from_date, to_date), sep=',')

    print_errors()


def scrape_seasons(seasons, if_scrape_shifts):
    """
    Given list of seasons -> scrape all seasons
    :param seasons: list of seasons
    :param if_scrape_shifts: if scrape shifts
    :return: nothing
    """
    for season in seasons:
        from_date = '-'.join([str(season), '10', '1'])
        to_date = '-'.join([str(season + 1), '7', '01'])

        games = json_schedule.scrape_schedule(from_date, to_date)
        pbp_df, shifts_df = scrape_list_of_games(games, if_scrape_shifts)

        if pbp_df is not None:
            #pbp_df.to_csv('nhl_pbp{}{}.csv'.format(season, season + 1), sep=',')
            pipeline_functions.write_boto_s3(pbp_df, 'shots-all', 'nhl_pbp{}{}.csv'.format(season, season + 1))
        if shifts_df is not None:
            #shifts_df.to_csv('nhl_shifts{}{}.csv'.format(season, season + 1), sep=',')
            pipeline_functions.write_boto_s3(shifts_df, 'shots-all', 'nhl_shifts{}{}.csv'.format(season, season + 1))

    print_errors()


def scrape_games(last_game, last_date, szn, if_scrape_shifts, process = True, model = False):
    """
    Scrape a given game
    :param games: list of game_ids
    :param if_scrape_shifts: if scrape shifts
    :return: nothing
    """
    import numpy as np

    pbp_types = {'Game_Id': int,
                 'Date': object,
                 'Period': int,
                 'Event': object,
                 'Description': object,
                 'Time_Elapsed': object,
                 'Seconds_Elapsed': int,
                 'Strength': object,
                 'Ev_Zone': object,
                 'Type': object,
                 'Ev_Team': object,
                 'Home_Zone': object,
                 'Away_Team': object,
                 'Home_Team': object,
                 'xC': np.float64,
                 'yC': np.float64,
                 'Home_Coach': object,
                 'Away_Coach': object
                 }

    last_season = str(szn - 1) + str(szn)
    this_season = str(szn) + str(szn + 1)

    ## Read last 2 seasons of data
    pbp_df_t_1 = pipeline_functions.encode_data(pipeline_functions.read_boto_s3('shots-all', 'nhl_pbp' + str(last_season) + '.csv'), types = pbp_types)
    pbp_df_t0 = pipeline_functions.encode_data(pipeline_functions.read_boto_s3('shots-all', 'nhl_pbp' + str(this_season) + '.csv'), types = pbp_types)

    ## Append last 2 seasons
    pbp_df_all = pbp_df_t_1.append(pbp_df_t0)

    prior_games = max(pbp_df_t0.Game_Id)
    print(prior_games)

    # shifts_df_all = pd.read_csv("nhl_shifts" + str(this_season) + ".csv", encoding='latin-1')
    shifts_df_all = pipeline_functions.read_boto_s3('shots-all', 'nhl_shifts' + str(this_season) + '.csv')

    if last_date is None:
        prior_games = max(pbp_df_t0.Game_Id)
        print(prior_games)
        games = list(range(prior_games + 1, last_game + 1))
        games_list = list(map(str, games))
        games_list = [str(szn) + '0' + x for x in games_list]
        games_list = list(map(int, games_list))
        ## Create List of game_id's and dates
        games_list = json_schedule.get_dates(games_list)
    else:
        prior_dates = max(pbp_df_t0.Date)
        print(prior_dates)
        games_list = json_schedule.scrape_schedule(prior_dates,last_date)

    ## Update Rosters
    #df_adjustments.roster_info_update(str(szn)+str(szn+1))

    print(games_list)

    pbp_df, shifts_df = scrape_list_of_games(games_list, if_scrape_shifts)
    pbp_df = pipeline_functions.encode_data(pbp_df, types = pbp_types)
    shifts_df = pipeline_functions.encode_data(pbp_df)

    # Save/Load/Append/Save PBP
    pbp_df = pipeline_functions.encode_data(pbp_df)
    print("Total game count: " + str(pbp_df.loc[:,['Game_Id']].drop_duplicates().count()))

    pbp_df_updated = pbp_df_t0.append(pbp_df).drop_duplicates()
    pbp_df_updated['Date'] = pd.to_datetime(pbp_df_updated['Date']).dt.date
    pipeline_functions.write_boto_s3(pbp_df_updated, 'shots-all', 'nhl_pbp' + str(this_season) + '.csv')

    # Save/Load/Append/Save Shifts
    #shifts_df.to_csv('nhl_shifts.csv', index=False)
    #shifts_df = pd.read_csv("nhl_shifts.csv", encoding='latin-1')
    shifts_df_all = shifts_df_all.append(shifts_df).drop_duplicates()
    pipeline_functions.write_boto_s3(shifts_df_all, 'shots-all', 'nhl_shifts' + str(this_season) + '.csv')

    if process == True:
        print('Data Prep')

        pbp_df = pbp_df_all.append(pbp_df)

        pbp_df = df_adjustments.transform_data(pbp_df)
        pbp_df = df_adjustments.lookups_data_clean(pbp_df)
        pbp_df = df_adjustments.cumulative_shooting_talent(pbp_df)
        model_df = df_adjustments.feature_generation(pbp_df)

        pbp_df['season_model'] = pbp_df.apply(lambda x: '2011_2012' if x.season in ['20102011', '20112012'] else
        '2013_2014' if x.season in ['20122013', '20132014'] else
        '2015_2016' if x.season in ['20142015', '20152016'] else
        '2017_2019' if x.season in ['20162017', '20172018', '20182019'] else 0, axis=1)

        model_df['season_model'] = model_df.apply(lambda x: '2011_2012' if x.season in ['20102011', '20112012'] else
        '2013_2014' if x.season in ['20122013', '20132014'] else
        '2015_2016' if x.season in ['20142015', '20152016'] else
        '2017_2019' if x.season in ['20162017', '20172018', '20182019'] else 0, axis=1)

    if model == False:
        print('Score Only')
        df_adjustments.All_Model_ScoringOnly(model_df, pbp_df, '2017_2019')
    else:
        print('Model & Score')
        df_adjustments.All_Model_Scoring(model_df, pbp_df, '2017_2019')

    print_errors()


def print_errors():
    """
    Print errors with scraping
    """
    global broken_shifts_games
    global broken_pbp_games
    global players_missing_ids
    global espn_games

    print('\nBroken pbp:')
    for x in broken_pbp_games:
        print(x)

    print('Broken shifts:')
    for x in broken_shifts_games:
        print(x)

    print('Missing ids:')
    global players_missing_ids
    for x in players_missing_ids:
        print(x)

    print('ESPN games:')
    for x in espn_games:
        print(x)

    broken_shifts_games = []
    broken_pbp_games = []
    players_missing_ids = []
    espn_games = []






