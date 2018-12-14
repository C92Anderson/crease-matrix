import json_pbp
import html_pbp
import espn_pbp
import shared
import requests
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import json_shifts
import html_shifts
import playing_roster
import json_schedule
import pandas as pd
import boto3
import time
import datetime
import thinkbayes2 as tb
import pipeline_functions
import io
import pickle

# Additional Functions
def s3_model_object_load(bucket, location):
    s3 = boto3.resource('s3')
    with io.BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(location, data)
        data.seek(0)  # move back to the beginning after writing
        m = pickle.load(data)
    return(m)

def s3_model_object_dump(model, bucket, filename):
    s3_resource = boto3.resource('s3')
    pickle_byte_obj = pickle.dumps(model)
    s3_resource.Object(bucket, 'Models/' + str(filename)).put(Body=pickle_byte_obj)

# Create object to adjust for rink bias

class RinkAdjust(object):

    def __init__(self):
        self.teamxcdf, self.teamycdf, self.otherxcdf, self.otherycdf = {}, {}, {}, {}

    def addCDFs(self, team, this_x_cdf, this_y_cdf, other_x_cdf, other_y_cdf):
        self.teamxcdf[team] = this_x_cdf
        self.teamycdf[team] = this_y_cdf
        self.otherxcdf[team] = other_x_cdf
        self.otherycdf[team] = other_y_cdf

    def addTeam(self, team, this_team, rest_of_league):
        this_x_cdf = tb.MakeCdfFromPmf(tb.MakePmfFromList(this_team.X_unadj))
        this_y_cdf = tb.MakeCdfFromPmf(tb.MakePmfFromList(this_team.Y_unadj))
        other_x_cdf = tb.MakeCdfFromPmf(tb.MakePmfFromList(rest_of_league.X_unadj))
        other_y_cdf = tb.MakeCdfFromPmf(tb.MakePmfFromList(rest_of_league.Y_unadj))
        self.addCDFs(team, this_x_cdf, this_y_cdf, other_x_cdf, other_y_cdf)

    def PlotTeamCDFs(self, team, savefig=False):
        this_x_cdf = self.teamxcdf[team]
        this_y_cdf = self.teamycdf[team]
        other_x_cdf = self.otherxcdf[team]
        other_y_cdf = self.otherycdf[team]

        f, axx = plt.subplots(1, 2, sharey='col')
        f.set_size_inches(14, 8)

        xx1, yx1 = this_x_cdf.Render()
        xx2, yx2 = other_x_cdf.Render()

        axx[0].plot(xx1, yx1, color='blue', label='@%s' % team)
        axx[0].plot(xx2, yx2, color='brown', label='@Rest of League')
        axx[0].set_xlabel('CDF of X')
        axx[0].legend()

        xy1, yy1 = this_y_cdf.Render()
        xy2, yy2 = other_y_cdf.Render()

        axx[1].plot(xy1, yy1, color='blue', label='@%s' % team)
        axx[1].plot(xy2, yy2, color='brown', label='@Rest of League')
        axx[1].set_xlabel('CDF of Y')
        axx[1].legend()

        f.suptitle('Cumulative Density Function for Shot Location Rink Bias Adjustment')

        plt.show()

        if savefig:
            # f.set_tight_layout( True )
            plt.savefig('Rink bias CDF chart %s.png' % team)

    def rink_bias_adjust(self, x, y, team):
        """ this method implements the actual location conversion from biased to "unbiased" shot location

         the way it works for rink bias adjustment is that for a given shot location in a specific rink,
         you find the cumulative probabilities for that x and y in that rink. Then you calculate the league
         equivalent x and y that have the same probabilities as the one measured in the specific rink

         The equivalency CDFs are calculated using only visiting teams, which ensures that both single rink and
         league wide rinks have as wide a sample of teams as possible but avoid any possible home team bias.
         All of which lets us assume that they are then unbiased enough to be representative (at least enough
         for standardization purposes)

         This is (my adaption of my understanding of) Shuckers' method for rink bias adjustment as described in Appendix A here:
         http://www.sloansportsconference.com/wp-content/uploads/2013/Total%20Hockey%20Rating%20(THoR)%20A%20comprehensive%20statistical%20rating%20of%20National%20Hockey%20League%20forwards%20and%20defensemen%20based%20upon%20all%20on-ice%20events.pdf

         for example, if a shot x coordinate is measured as xmeas in a rink

             xprob = this_x_cdf.Prob( xmeas )  # cum prob of seeing xmeas in this rink
             xadj = other_x_cdf.Value( xprob ) # value associated with same prob in rest of league

        analogous process for y

        The code for Cdf/Pmf creation and manipulation is taken directly from Allan Downey's code for "Think Bayes"
        """

        xprob = self.teamxcdf[team].Prob(x)
        newx = self.otherxcdf[team].Value(xprob)

        yprob = self.teamycdf[team].Prob(y)
        newy = self.otherycdf[team].Value(yprob)

        return newx, newy


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * (180 / np.pi)


# Process data and subset to shots, adjusting for rink bias
def transform_data(data):
    import warnings
    warnings.simplefilter("ignore")

    from sqlalchemy import create_engine
    engine = create_engine(
        'mysql+mysqlconnector://cole92anderson:cprice31!@css-db.cnqvzrgc2pnj.us-east-1.rds.amazonaws.com:3306/nhl_all')

    pbp_df = data

    print("All events and columns: " + str(pbp_df.shape))

    ## Remove shootouts
    pbp_df['season'] = pbp_df.apply(
        lambda x: str(pd.to_datetime(x.Date).year - 1) + str(pd.to_datetime(x.Date).year) if pd.to_datetime(
            x.Date).month < 9 else str(pd.to_datetime(x.Date).year) + str(pd.to_datetime(x.Date).year + 1), axis=1)

    pbp_df['season2'] = pbp_df.apply(lambda x: x.season if x.Game_Id < 30000 else str(x.season) + "p", axis=1)

    pbp_df['Season_Type'] = pbp_df.apply(lambda x: 'RS' if x.Game_Id < 30000 else 'PO', axis=1)

    pbp_df['season_model'] = pbp_df.apply(lambda x: '2011_2012' if x.season in ['20102011', '20112012'] else
    '2013_2014' if x.season in ['20122013', '20132014'] else
    '2015_2016' if x.season in ['20142015', '20152016'] else
    '2017_2018' if x.season in ['20162017', '20172018'] else 0, axis=1)

    ## Drop Duplicates
    pbp_df = pbp_df.drop_duplicates(['season', 'Game_Id', 'Period', 'Ev_Team', 'Seconds_Elapsed', 'Event'])

    pbp_df = pbp_df.sort_values(['season', 'Game_Id', 'Period', 'Seconds_Elapsed'], ascending=True)

    # Remove SOs
    pbp_df = pbp_df.loc[((pbp_df.Period == 5) & (pbp_df.Season_Type == "RS")) != True, :]

    # Group Give/Take together
    pbp_df['Event'] = pbp_df['Event'].apply(lambda x: 'TURN' if x in ["GIVE", "TAKE"] else x)

    pbp_df['Type'] = pbp_df['Type'].apply(lambda x: 'DEFLECTED' if x in ["DEFLECTED", "TIP-IN"] else \
        'WRIST SHOT' if x in ["WRIST SHOT", "SNAP SHOT", "WRAP-AROUND"] else x)

    ## Check Lag Time doesn't Cross Periods
    pbp_df = pbp_df.sort_values(['season', 'Game_Id', 'Period', 'Seconds_Elapsed'], ascending=True)

    pbp_df['lagged_Event'] = pbp_df.groupby(['season', 'Game_Id', 'Period'])['Event'].shift(1)
    pbp_df['lagged_Ev_Zone'] = pbp_df.groupby(['season', 'Game_Id', 'Period'])['Ev_Zone'].shift(1)
    pbp_df['lagged_Seconds_Elapsed'] = pbp_df.groupby(['season', 'Game_Id', 'Period'])['Seconds_Elapsed'].shift(1)

    pbp_df['lagged_xC'] = pbp_df.groupby(['season', 'Game_Id', 'Period'])['xC'].shift(1)
    pbp_df['lagged_yC'] = pbp_df.groupby(['season', 'Game_Id', 'Period'])['yC'].shift(1)

    pbp_df['Away_Score'] = pbp_df.groupby(['season', 'Game_Id'])['Away_Score'].shift(1)
    pbp_df['Home_Score'] = pbp_df.groupby(['season', 'Game_Id'])['Home_Score'].shift(1)

    #############################################
    ### Subset to just shots
    #############################################
    pbp_df = pbp_df.loc[pbp_df.Event.isin(["SHOT", "GOAL", "MISS", "BLOCK"]), :]

    print("All shots/blocks and columns: " + str(pbp_df.shape))

    ## Binary
    pbp_df['Goal'] = pbp_df.apply(lambda x: 1 if x.Event == "GOAL" else 0, axis=1)

    pbp_df['EmptyNet_SA'] = pbp_df.apply(lambda x: 1 if ((pd.isnull(x.Home_Goalie)) & (x.Ev_Team == x.Away_Team)) | \
                                                        ((pd.isnull(x.Away_Goalie)) & (
                                                                    x.Ev_Team == x.Home_Team)) else 0, axis=1)

    pbp_df['is_Rebound'] = pbp_df.apply(lambda x: 1 if (x.lagged_Event in ["SHOT"]) & \
                                                       ((x.Seconds_Elapsed - x.lagged_Seconds_Elapsed) <= 2) else 0,
                                        axis=1)

    pbp_df['is_Bounce'] = pbp_df.apply(lambda x: 1 if (x.lagged_Event in ["BLOCK", "MISS"]) & \
                                                      ((x.Seconds_Elapsed - x.lagged_Seconds_Elapsed) <= 2) else 0,
                                       axis=1)

    pbp_df['is_Rush'] = pbp_df.apply(lambda x: 1 if (x.Ev_Zone != x.lagged_Ev_Zone) & \
                                                    ((x.Seconds_Elapsed - x.lagged_Seconds_Elapsed) <= 6) else 0,
                                     axis=1)

    # Replace every occurrence of PHX with ARI
    pbp_df['Home_Team'] = pbp_df.apply(lambda x: x.Home_Team if x.Home_Team != 'PHX' else 'ARI', axis=1)
    pbp_df['Away_Team'] = pbp_df.apply(lambda x: x.Away_Team if x.Away_Team != 'PHX' else 'ARI', axis=1)
    pbp_df['Ev_Team'] = pbp_df.apply(lambda x: x.Ev_Team if x.Ev_Team != 'PHX' else 'ARI', axis=1)
    # Replace every occurrence of ATL with WPG
    pbp_df['Home_Team'] = pbp_df.apply(lambda x: x.Home_Team if x.Home_Team != 'ATL' else 'WPG', axis=1)
    pbp_df['Away_Team'] = pbp_df.apply(lambda x: x.Away_Team if x.Away_Team != 'ATL' else 'WPG', axis=1)
    pbp_df['Ev_Team'] = pbp_df.apply(lambda x: x.Ev_Team if x.Ev_Team != 'ATL' else 'WPG', axis=1)

    # add a 'Direction' column to indicate the primary direction for shots. The heuristic to determine
    # direction is the sign of the median of the X coordinate of shots in each period. This then lets us filter
    # out shots that originate from back in the defensive zone when the signs don't match
    pbp_df['Home_Shooter'] = pbp_df.apply(lambda x: 1 if x.Ev_Team == x.Home_Team else 0, axis=1)

    game_period_locations = pbp_df.groupby(by=['season', 'Game_Id', 'Period', 'Home_Shooter'])['xC', 'yC']

    game_period_medians = game_period_locations.transform(np.median)

    pbp_df['Direction'] = np.sign(game_period_medians['xC'])

    # should actually write this to a CSV as up to here is the performance intensive part
    pbp_df['X_unadj'], pbp_df['Y_unadj'] = zip(
        *pbp_df.apply(lambda x: (x.xC, x.yC) if x.Direction > 0 else (-x.xC, -x.yC), axis=1))
    pbp_df['lagged_xC'], pbp_df['lagged_yC'] = zip(
        *pbp_df.apply(lambda x: (x.lagged_xC, x.lagged_yC) if x.Direction > 0 else (-x.lagged_xC, -x.lagged_yC),
                      axis=1))

    pbp_df['LS_Shot'] = pbp_df.apply(lambda x: 1 if x.Y_unadj < 0 else 0, axis=1)

    ## Logged Last Event Time
    pbp_df['LN_Last_Event_Time'] = pbp_df.apply(lambda x: 0 if (x.Seconds_Elapsed - x.lagged_Seconds_Elapsed) <= 0 \
        else np.log(x.Seconds_Elapsed - x.lagged_Seconds_Elapsed + 0.001), axis=1)

    # Last Event
    pbp_df['LastEV_Off_Faceoff'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Off') & (x.lagged_Event == 'FAC') else 0, axis=1)
    pbp_df['LastEV_Def_Faceoff'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Def') & (x.lagged_Event == 'FAC') else 0, axis=1)
    pbp_df['LastEV_Neu_Faceoff'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Neu') & (x.lagged_Event == 'FAC') else 0, axis=1)
    pbp_df['LastEV_Off_Shot'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Off') & (x.lagged_Event in ["SHOT", "MISS", "BLOCK"]) else 0,
        axis=1)
    pbp_df['LastEV_Def_Shot'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Def') & (x.lagged_Event in ["SHOT", "MISS", "BLOCK"]) else 0,
        axis=1)
    pbp_df['LastEV_Neu_Shot'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Neu') & (x.lagged_Event in ["SHOT", "MISS", "BLOCK"]) else 0,
        axis=1)
    pbp_df['LastEV_Off_Give'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Off') & (x.lagged_Event == 'TURN') else 0, axis=1)
    pbp_df['LastEV_Def_Give'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Def') & (x.lagged_Event == 'TURN') else 0, axis=1)
    pbp_df['LastEV_Neu_Give'] = pbp_df.apply(
        lambda x: x.LN_Last_Event_Time if (x.Ev_Zone == 'Neu') & (x.lagged_Event == 'TURN') else 0, axis=1)

    # Last Event Distance Change
    pbp_df['LastEV_Distance'] = np.sqrt(
        ((pbp_df['lagged_xC'] - pbp_df['X_unadj']) ** 2) + ((pbp_df['lagged_yC'] - pbp_df['Y_unadj']) ** 2))

    pbp_df['LastEV_FtperSec_Change'] = pbp_df.apply(lambda x: 0 if (x.Seconds_Elapsed - x.lagged_Seconds_Elapsed) <= 0 \
        else x.LastEV_Distance / (x.Seconds_Elapsed - x.lagged_Seconds_Elapsed),
                                                    axis=1)

    # Last Event Angle Change
    pbp_df['LastEV_Angle_Change'] = pbp_df.apply(
        lambda x: angle_between((x.Y_unadj, 89 - x.X_unadj), (x.lagged_yC, 89 - x.lagged_xC)) \
            if angle_between((x.Y_unadj, 89 - x.X_unadj), (x.lagged_yC, 89 - x.lagged_xC)) > -1
        else 0, axis=1)

    pbp_df['LastEV_AngleperSec_Change'] = pbp_df.apply(
        lambda x: 0 if (x.Seconds_Elapsed - x.lagged_Seconds_Elapsed) <= 0 \
            else x.LastEV_Angle_Change / (x.Seconds_Elapsed - x.lagged_Seconds_Elapsed),
        axis=1)

    ## Logged Changes
    pbp_df['LN_LastEV_AngleperFt_Change'] = pbp_df.apply(
        lambda x: np.log(0.0001) if x.LastEV_Distance <= 0 or x.LastEV_Angle_Change <= 0 \
            else np.log((x.LastEV_Angle_Change / x.LastEV_Distance) + 0.0001), axis=1)

    pbp_df['LN_LastEV_FtperSec_Change'] = pbp_df.apply(
        lambda x: np.log(0.0001) if x.LastEV_FtperSec_Change <= 0 or np.isnan(np.log(x.LastEV_FtperSec_Change)) \
            else np.log(x.LastEV_FtperSec_Change + 0.0001), axis=1)
    pbp_df['LN_LastEV_AngleperSec_Change'] = pbp_df.apply(
        lambda x: np.log(0.0001) if x.LastEV_AngleperSec_Change <= 0 or np.isnan(np.log(x.LastEV_AngleperSec_Change)) \
            else np.log(x.LastEV_AngleperSec_Change + 0.0001), axis=1)

    ## Adjust X, Y coordinates by Rink, using CDF of shot attempts only (remove blocks since they skew data)
    pbp_df = pbp_df.loc[pbp_df.Event.isin(["SHOT", "GOAL", "MISS"]), :]

    ## Call RinkAdjust class
    adjuster = RinkAdjust()

    ## New dataframe of adjusted shots for each home rink
    pbp_df_adj = pd.DataFrame()

    ## For each home rink
    for team in sorted(pbp_df.Home_Team.unique()):

        ## Split shots into team arena and all other rinks
        shot_data = pbp_df
        rink_shots = shot_data[shot_data.Home_Team == team]
        rest_of_league = shot_data[shot_data.Home_Team != team]

        ## Create teamxcdf and otherxcdf for rink adjustment
        adjuster.addTeam(team, rink_shots, rest_of_league)

        ## Adjusted coordinates
        Xadj = []
        Yadj = []

        ## For each shot in rink adjust coordinates based on other rinks
        for row in rink_shots.itertuples():
            newx, newy = adjuster.rink_bias_adjust(row.X_unadj, row.Y_unadj, row.Home_Team)

            Xadj.append(newx)
            Yadj.append(newy)

        rink_shots['X'] = Xadj
        rink_shots['Y'] = Yadj

        pbp_df_adj = pbp_df_adj.append(rink_shots)

    print ("All shots columns, rink adjusted: " + str(pbp_df_adj.shape))

    ## Apply only to season level data after x,y CDF adjustment
    pbp_df_adj['Shot_Distance_Unadj'] = pbp_df_adj.apply(lambda x: ((89 - x.X_unadj) ** 2 + (x.Y_unadj ** 2)) ** 0.5,
                                                         axis=1)
    pbp_df_adj['Shot_Distance'] = pbp_df_adj.apply(lambda x: ((89 - x.X) ** 2 + (x.Y ** 2)) ** 0.5, axis=1)
    pbp_df_adj['Shot_Angle'] = pbp_df_adj.apply(
        lambda x: np.arctan(abs(89 - x.X) / abs(0 - x.Y)) * (180 / np.pi) if x.Y != 0 \
            else 90, axis=1)

    pbp_df_adj['Last_Shot_Distance'] = pbp_df_adj.groupby(['season', 'Game_Id', 'Period', 'Home_Shooter'])[
        'Shot_Distance'].shift(1)
    pbp_df_adj['Last_Shot_Angle'] = pbp_df_adj.groupby(['season', 'Game_Id', 'Period', 'Home_Shooter'])[
        'Shot_Angle'].shift(1)
    pbp_df_adj['Last_LS_Shot'] = pbp_df_adj.groupby(['season', 'Game_Id', 'Period', 'Home_Shooter'])['LS_Shot'].shift(1)

    pbp_df_adj['Rebound_Distance_Change'] = pbp_df_adj.apply(
        lambda x: x.Last_Shot_Distance + x.Shot_Distance if x.is_Rebound == 1 else 0, axis=1)
    pbp_df_adj['Rebound_Angle_Change'] = pbp_df_adj.apply(lambda x: 0 if x.is_Rebound == 0 \
        else abs(x.Last_Shot_Angle - x.Shot_Angle) \
        if x.is_Rebound == 1 & (x.Last_LS_Shot == x.LS_Shot) else \
        (180 - x.Last_Shot_Angle - x.Shot_Angle), axis=1)

    pbp_df_adj['Rebound_Angular_Velocity'] = pbp_df_adj. \
        apply(lambda x: x.Rebound_Angle_Change / x.Rebound_Distance_Change \
        if x.Rebound_Distance_Change > 0 else 0, axis=1)

    pbp_df_adj['LN_Rebound_Angular_Velocity'] = pbp_df_adj. \
        apply(lambda x: np.log(x.Rebound_Angular_Velocity) \
        if x.Rebound_Angular_Velocity > 0 else 0, axis=1)

    print ("All shots columns, final calcuations: " + str(pbp_df_adj.shape))

    return pbp_df_adj


def lookups_data_clean(data):
    player_lookup = pipeline_functions.read_boto_s3('hockey-all', 'hockey_roster_info.csv')
    print("Player Lookup Dim: " + str(player_lookup.shape))
    player_lookup = player_lookup.sort_values('seasonId', ascending=False).groupby(['playerId']).first().reset_index(). \
                        loc[:, ['playerBirthDate', 'playerPositionCode', 'playerShootsCatches', 'playerId']]

    skater_lookup = player_lookup.loc[player_lookup.playerPositionCode != "G", :]
    skater_lookup.columns = ['shooterDOB', 'Player_Position', 'Shoots', 'p1_ID']
    skater_lookup['p1_ID'] = skater_lookup['p1_ID'].astype(str)

    goalie_lookup = player_lookup.loc[player_lookup.playerPositionCode == "G", :]
    goalie_lookup = goalie_lookup.rename(index=str,
                                         columns={"playerId": "SA_Goalie_Id", "playerShootsCatches": "Catches",
                                                  "playerBirthDate": "goalieDOB"})

    goalie_lookup['SA_Goalie_Id'] = goalie_lookup['SA_Goalie_Id'].astype(str)

    for col in ['Game_Id', 'Away_Goalie_Id', 'Home_Goalie_Id', 'p1_ID', 'p2_ID', 'p3_ID',
                'awayPlayer1_id', 'awayPlayer2_id', 'awayPlayer3_id', 'awayPlayer4_id', 'awayPlayer5_id',
                'awayPlayer6_id',
                'homePlayer1_id', 'homePlayer2_id', 'homePlayer3_id', 'homePlayer4_id', 'homePlayer5_id',
                'homePlayer6_id']:
        data[col] = data[col].fillna(0).astype(int).astype(str)

    data['SA_Goalie'] = data.apply(lambda x: x.Away_Goalie if x.Ev_Team == x.Home_Team else x.Home_Goalie, axis=1)
    data['SA_Goalie_Id'] = data.apply(lambda x: x.Away_Goalie_Id if x.Ev_Team == x.Home_Team else x.Home_Goalie_Id,
                                      axis=1)

    data['Away_State'] = data.apply(
        lambda x: x.Away_Players - 1 if x.Away_Goalie_Id in [x.awayPlayer6_id, x.awayPlayer5_id, x.awayPlayer4_id,
                                                             x.awayPlayer3_id] else x.Away_Players, axis=1)
    data['Home_State'] = data.apply(
        lambda x: x.Home_Players - 1 if x.Home_Goalie_Id in [x.homePlayer6_id, x.homePlayer5_id, x.homePlayer4_id,
                                                             x.homePlayer3_id] else x.Home_Players, axis=1)

    data['Away_State'] = data.apply(
        lambda x: x.Away_Players - 1 if x.Away_Goalie_Id in [x.awayPlayer6_id, x.awayPlayer5_id, x.awayPlayer4_id,
                                                             x.awayPlayer3_id] else x.Away_Players, axis=1)
    data['Home_State'] = data.apply(
        lambda x: x.Home_Players - 1 if x.Home_Goalie_Id in [x.homePlayer6_id, x.homePlayer5_id, x.homePlayer4_id,
                                                             x.homePlayer3_id] else x.Home_Players, axis=1)

    data['Results_inRebound'] = data['is_Rebound'].shift(periods=-1)

    data['Shooter_State'] = data.apply(lambda x: x.Away_State if x.Ev_Team != x.Home_Team else x.Home_State, axis=1)
    data['Goalie_State'] = data.apply(lambda x: x.Away_State if x.Ev_Team == x.Home_Team else x.Home_State, axis=1)

    data['Game_State'] = data.apply(lambda x: str(x.Away_State) + "v" + str(x.Home_State) if x.Ev_Team != x.Home_Team \
        else str(x.Home_State) + "v" + str(x.Away_State), axis=1)
    data['Game_State'] = data.apply(lambda x: "SH_SF" if x.Game_State in ["3v5", "3v4", "3v6", "4v5", "4v6", "5v6"] else \
        "PP_2p_SF" if x.Game_State in ["6v3", "6v4", "5v3"] else \
            "5v5" if x.Game_State in ["5v5", "6v6"] else \
                x.Game_State if x.Game_State in ["3v3", "4v4", "5v4", "4v3"] else \
                    "6v5" if x.Game_State in ["6v5", "7v5"] else "Other", axis=1)

    data['State_Space'] = data['Goalie_State'] + data['Shooter_State']
    data['Shooter_State_Advantage'] = data['Shooter_State'] - data['Goalie_State']

    data = data.merge(skater_lookup, on=['p1_ID'], how='left')
    data = data.merge(goalie_lookup, on=['SA_Goalie_Id'], how='left')

    data['Shooter_Handedness'] = data.apply(lambda x: "L" if x.Shoots == "L" else \
        "R" if x.Shoots == "R" else "U", axis=1)

    data['Handed_Class'] = data['Shoots'].str.cat(data['Catches'], sep='')

    data['Handed_Class2'] = data.apply(lambda x: "Same" if x.Handed_Class in ["LL", "RR"] else \
        "Opposite" if x.Handed_Class in ["LR", "RL"] else "U", axis=1)

    data['Player_Position2'] = data.apply(lambda x: "D" if x.Player_Position == "D" else "F", axis=1)

    return data


def cumulative_shooting_talent(data):
    shooting_percentage = data.groupby(['Player_Position2'])['Goal'].mean()

    data['Cum_Goal'] = data.groupby(['p1_ID'])['Goal'].cumsum()
    data['Cum_Shots'] = data.groupby(['p1_ID']).cumcount()

    data['Cum_Goal'] = data.apply(lambda x: x.Cum_Goal - 1 if x.Event == "GOAL" else x.Cum_Goal, axis=1)

    kr21_stabilizer_F = pd.to_numeric(375.0)
    kr21_stabilizer_D = pd.to_numeric(275.0)

    data['Regressed_Shooting_Indexed'] = data.apply(
        lambda x: ((x.Cum_Goal + (kr21_stabilizer_D * shooting_percentage[0])) / \
                   (x.Cum_Shots + kr21_stabilizer_D)) / shooting_percentage[0] \
            if x.Player_Position2 == "D" else ((x.Cum_Goal + (kr21_stabilizer_F * shooting_percentage[1])) / \
                                               (x.Cum_Shots + kr21_stabilizer_F)) / shooting_percentage[1], axis=1)

    return data


def feature_generation(data,
                       id_vars=["season"],
                       target_vars=['Goal', 'Results_inRebound'],
                       num_vars=["EmptyNet_SA", "is_Rebound", "is_Rush", "LN_Last_Event_Time", "LastEV_Off_Faceoff",
                                 "LastEV_Def_Faceoff", "LastEV_Neu_Faceoff", "LastEV_Off_Shot", "LastEV_Def_Shot",
                                 "LastEV_Neu_Shot",
                                 "LastEV_Off_Give", "LastEV_Def_Give", "LastEV_Neu_Give", "LN_Rebound_Angular_Velocity",
                                 "LN_LastEV_FtperSec_Change", "LN_LastEV_AngleperSec_Change",
                                 "LN_LastEV_AngleperFt_Change",
                                 "Regressed_Shooting_Indexed"],
                       cat_vars=["Type", "Game_State", "Handed_Class2", "Player_Position2"],
                       poly_vars=["Shot_Distance", "Shot_Angle"],
                       model_vars=['EmptyNet_SA', 'is_Rebound', 'is_Rush', 'LN_Last_Event_Time',
                                   'LastEV_Off_Faceoff', 'LastEV_Def_Faceoff', 'LastEV_Neu_Faceoff',
                                   'LastEV_Off_Shot', 'LastEV_Def_Shot', 'LastEV_Neu_Shot',
                                   'LastEV_Off_Give', 'LastEV_Def_Give', 'LastEV_Neu_Give',
                                   'LN_LastEV_FtperSec_Change', 'LN_LastEV_AngleperSec_Change',
                                   'LN_LastEV_AngleperFt_Change',

                                   'LN_Rebound_Angular_Velocity', 'Regressed_Shooting_Indexed',
                                   'Type_BACKHAND', 'Type_DEFLECTED', 'Type_SLAP SHOT',
                                   'Type_WRIST SHOT',
                                   'Game_State_3v3', 'Game_State_4v3', 'Game_State_4v4', 'Game_State_5v5',
                                   'Game_State_5v4', 'Game_State_6v5', 'Game_State_Other', 'Game_State_PP_2p_SF',
                                   'Game_State_SH_SF',
                                   'Handed_Class2_Opposite',
                                   'Player_Position2_F', 'Shot_Distance',
                                   'Shot_Distance^2', 'Shot_Distance^3', 'Shot_Angle', 'Shot_Angle^2',
                                   'Shot_Angle^3']):
    from sklearn.preprocessing import PolynomialFeatures
    ## Dummy Variables
    model_data = data[num_vars].fillna(0)

    for i in cat_vars:
        var_dummies = pd.get_dummies(data.loc[:, [i]])

        model_data = pd.concat([model_data, var_dummies], axis=1)

    ## Polynomial Variables
    for i in poly_vars:
        poly_data = data.loc[:, [i]]

        poly = PolynomialFeatures(degree=3, interaction_only=False).fit(poly_data)
        poly_names = poly.get_feature_names(poly_data.columns)

        poly_output = poly.transform(data.loc[:, [i]])

        model_data = pd.DataFrame(pd.concat([model_data,
                                             pd.DataFrame(poly_output,
                                                          columns=poly_names).iloc[:, 1:]], axis=1))

    # model_mat = model_data.loc[:, model_vars].as_matrix()
    model_data = pd.concat([data[id_vars], data[target_vars], model_data], axis=1)

    print(model_data.shape)

    return model_data


def All_Model_Scoring(model_data, data, szn):
    print (szn)

    from sklearn.cross_validation import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import LogisticRegressionCV
    import pickle

    model_vars = ['EmptyNet_SA', 'is_Rebound', 'is_Rush', 'LN_Last_Event_Time',
                  'LastEV_Off_Faceoff', 'LastEV_Def_Faceoff', 'LastEV_Neu_Faceoff',
                  'LastEV_Off_Shot', 'LastEV_Def_Shot', 'LastEV_Neu_Shot',
                  'LN_LastEV_FtperSec_Change', 'LN_LastEV_AngleperFt_Change',
                  'LastEV_Off_Give', 'LastEV_Def_Give', 'LastEV_Neu_Give',
                  'LN_Rebound_Angular_Velocity', 'Regressed_Shooting_Indexed',
                  'Type_BACKHAND', 'Type_DEFLECTED', 'Type_SLAP SHOT',
                  'Type_WRIST SHOT',
                  'Game_State_3v3', 'Game_State_4v3', 'Game_State_4v4', 'Game_State_5v5', 'Game_State_5v4',
                  'Game_State_6v5', 'Game_State_Other', 'Game_State_PP_2p_SF', 'Game_State_SH_SF',
                  'Handed_Class2_Opposite',
                  'Player_Position2_F', 'Shot_Distance',
                  'Shot_Distance^2', 'Shot_Distance^3', 'Shot_Angle', 'Shot_Angle^2',
                  'Shot_Angle^3']

    rebound_vars = ['xG_raw'] + model_vars

    szn_data = data.loc[data.season_model == szn, :]

    szn_model_data = model_data.loc[model_data.season_model == szn, :].fillna(0)
    szn_model_mat = szn_model_data.loc[szn_model_data.season_model == szn, model_vars].as_matrix().astype(np.float)

    # import pickle
    # szn_xG_Model = pickle.load(open('xG_Model_' + str(szn) + '_obj.sav', 'rb'))
    # xG_raw = szn_xG_Model.predict_proba(szn_model_mat)[:,1]

    ### Train xG Model
    goal = szn_model_data.Goal
    print (str(szn) + ' seasons dimensions: ' + str(szn_model_mat.shape))
    print (str(szn) + ' seasons shooting%: ' + str(sum(goal) / len(goal)))

    fold = KFold(len(goal), n_folds=10, shuffle=True, random_state=777)

    xG_model_CV = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        , penalty='l2'
        , scoring='roc_auc'
        , cv=fold
        , random_state=777
        , max_iter=10000
        , fit_intercept=True
        , solver='newton-cg'
        , tol=10
    )

    xG_model_CV.fit(szn_model_mat, goal)

    ## Save Model
    filename = 'xG_Model_' + str(szn) + '_obj.sav'
    s3_model_object_dump(xG_model_CV, 'shots-all', filename)

    print (str(szn) + 'Max auc_roc:', xG_model_CV.scores_[1].max())

    xG_raw = xG_model_CV.predict_proba(szn_model_mat)[:, 1]

    print (str(szn) + ' seasons goals: ' + str(sum(goal)) + ', season xG: ' + str(sum(xG_raw)))

    ### Assemble data and train xRebound Model
    rebound = szn_model_data.Results_inRebound.fillna(0)
    print (str(szn) + ' goals scored: ' + str(sum(szn_data.Goal)))
    print (str(szn) + ' xG scored: ' + str(sum(xG_raw)))

    print (str(szn) + ' seasons dimensions: ' + str(szn_model_mat.shape))
    print (str(szn) + ' seasons rebound%: ' + str(sum(rebound) / len(rebound)))

    fold = KFold(len(rebound), n_folds=10, shuffle=True, random_state=777)

    szn_model_mat = pd.concat([szn_model_data.reset_index(drop=True),
                               pd.DataFrame(xG_raw, columns=['xG_raw']).reset_index(drop=True)], axis=1).loc[:,
                    rebound_vars].as_matrix()

    xR_model_CV = LogisticRegressionCV(
        Cs=list(np.power(10.0, np.arange(-10, 10)))
        , penalty='l2'
        , scoring='roc_auc'
        , cv=fold
        , random_state=777
        , max_iter=10000
        , fit_intercept=True
        , solver='newton-cg'
        , tol=10
    )

    xR_model_CV.fit(szn_model_mat, rebound)

    filename = 'xR_Model_' + str(szn) + '_obj.sav'
    s3_model_object_dump(xR_model_CV, 'shots-all', filename)

    print (str(szn) + ' Max auc_roc:', xR_model_CV.scores_[1].max())

    xR_raw = xR_model_CV.predict_proba(szn_model_mat)[:, 1]

    print (str(szn) + ' seasons rebounds: ' + str(sum(rebound)) + ', season xR: ' + str(sum(xR_raw)))

    coefs = pd.DataFrame(list(zip(np.array(rebound_vars), xR_model_CV.coef_.T)),
                         columns=['Variable', 'Coef']).sort_values(['Coef'], ascending=False)

    scored_data = pd.concat([
        pd.DataFrame(xG_raw, columns=['xG_raw']).reset_index(drop=True),
        pd.DataFrame(xR_raw, columns=['xR']).reset_index(drop=True),
        szn_data.reset_index(drop=True)
    ], axis=1)

    #scored_data.to_csv("scored_data" + str(szn) + ".csv", index=False)
    pipeline_functions.write_boto_s3(scored_data, 'shots-all', "scored_data" + str(szn) + ".csv")

    return coefs.T


def All_Model_ScoringOnly(model_data, data, szn):
    print (szn)

    from sklearn.cross_validation import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import LogisticRegressionCV
    import pickle

    model_vars = ['EmptyNet_SA', 'is_Rebound', 'is_Rush', 'LN_Last_Event_Time',
                  'LastEV_Off_Faceoff', 'LastEV_Def_Faceoff', 'LastEV_Neu_Faceoff',
                  'LastEV_Off_Shot', 'LastEV_Def_Shot', 'LastEV_Neu_Shot',
                  'LN_LastEV_FtperSec_Change', 'LN_LastEV_AngleperFt_Change',
                  'LastEV_Off_Give', 'LastEV_Def_Give', 'LastEV_Neu_Give',
                  'LN_Rebound_Angular_Velocity', 'Regressed_Shooting_Indexed',
                  'Type_BACKHAND', 'Type_DEFLECTED', 'Type_SLAP SHOT',
                  'Type_WRIST SHOT',
                  'Game_State_3v3', 'Game_State_4v3', 'Game_State_4v4', 'Game_State_5v5', 'Game_State_5v4',
                  'Game_State_6v5', 'Game_State_Other', 'Game_State_PP_2p_SF', 'Game_State_SH_SF',
                  'Handed_Class2_Opposite',
                  'Player_Position2_F', 'Shot_Distance',
                  'Shot_Distance^2', 'Shot_Distance^3', 'Shot_Angle', 'Shot_Angle^2',
                  'Shot_Angle^3']

    rebound_vars = ['xG_raw'] + model_vars

    szn_data = data.loc[data.season_model == szn, :]

    szn_model_data = model_data.loc[model_data.season_model == szn, :].fillna(0)
    szn_model_mat = szn_model_data.loc[szn_model_data.season_model == szn, model_vars].as_matrix().astype(np.float)

    ### Train xG Model
    goal = szn_model_data.Goal
    print (str(szn) + ' seasons dimensions: ' + str(szn_model_mat.shape))
    print (str(szn) + ' seasons shooting%: ' + str(sum(goal) / len(goal)))

    ## Load Model
    filename = 'xG_Model_' + str(szn) + '_obj.sav'
    xG_model_CV = s3_model_object_load('shots-all','Models/' + str(filename))

    ## Model Ability
    print (str(szn) + 'Max auc_roc:', xG_model_CV.scores_[1].max())

    ## Score Data
    xG_raw = xG_model_CV.predict_proba(szn_model_mat)[:, 1]

    print (str(szn) + ' seasons goals: ' + str(sum(goal)) + ', season xG: ' + str(sum(xG_raw)))

    ### Assemble data and train xRebound Model
    rebound = szn_model_data.Results_inRebound.fillna(0)
    print (str(szn) + ' goals scored: ' + str(sum(szn_data.Goal)))
    print (str(szn) + ' xG scored: ' + str(sum(xG_raw)))

    print (str(szn) + ' seasons dimensions: ' + str(szn_model_mat.shape))
    print (str(szn) + ' seasons rebound%: ' + str(sum(rebound) / len(rebound)))

    ## Add xG to dataset
    szn_model_mat = pd.concat([szn_model_data.reset_index(drop=True),
                               pd.DataFrame(xG_raw, columns=['xG_raw']).reset_index(drop=True)], axis=1).loc[:,
                    rebound_vars].as_matrix()

    ## Load Model
    filename = 'xR_Model_' + str(szn) + '_obj.sav'
    xR_model_CV = s3_model_object_load('shots-all','Models/' + str(filename))

    print (str(szn) + ' Max auc_roc:', xR_model_CV.scores_[1].max())

    xR_raw = xR_model_CV.predict_proba(szn_model_mat)[:, 1]

    print (str(szn) + ' seasons rebounds: ' + str(sum(rebound)) + ', season xR: ' + str(sum(xR_raw)))

    coefs = pd.DataFrame(list(zip(np.array(rebound_vars), xR_model_CV.coef_.T)),
                         columns=['Variable', 'Coef']).sort_values(['Coef'], ascending=False)

    scored_data = pd.concat([
        pd.DataFrame(xG_raw, columns=['xG_raw']).reset_index(drop=True),
        pd.DataFrame(xR_raw, columns=['xR']).reset_index(drop=True),
        szn_data.reset_index(drop=True)
    ], axis=1)

    #scored_data.to_csv("scored_data" + str(szn) + ".csv", index=False)
    pipeline_functions.write_boto_s3(scored_data, 'shots-all', "scored_data" + str(szn) + ".csv")

    return coefs.T


def roster_update(season, games):
    import requests
    from datetime import datetime

    roster_master = pd.DataFrame()

    for i in games:

        try:
            url = "https://statsapi.web.nhl.com/api/v1/game/" + str(season) + "0" + str(i) + "/feed/live?site=en_nhl"

            # Read game
            data = requests.get(url).json()

            # Subset to roster
            players = data['gameData']['players']
            # print(type(players)) #dict

            # Check to see game exists
            if len(players) > 0:

                # Find list of player ids
                playerid_list = list(players.keys())

                # Shell game roster
                game_rosters = pd.DataFrame()

                # For each player return information
                for id in playerid_list:

                    # subset
                    player_info = pd.DataFrame(data['gameData']['players'][id])

                    player_info_dedup = player_info.loc[:,
                                        [u'active', u'alternateCaptain', u'birthCity', u'birthCountry',
                                         u'birthDate', u'birthStateProvince', u'captain', u'currentAge', u'firstName',
                                         u'fullName', u'height', u'id',
                                         u'lastName', u'link', u'nationality', u'primaryNumber', u'rookie',
                                         u'rosterStatus', u'shootsCatches',
                                         u'weight']].drop_duplicates()

                    # Find current team
                    try:
                        player_info_dedup['currentTeam'] = player_info.loc[['triCode'], ['currentTeam']].iloc[0, 0]
                    except KeyError:
                        continue

                    # Find current position
                    player_info_dedup['primaryPosition'] = player_info.loc[['abbreviation'], ['primaryPosition']].iloc[
                        0, 0]

                    ###append to master roster list
                    game_rosters = game_rosters.append(player_info_dedup)

                # Set index and nhl_id
                game_rosters['nhl_id'] = game_rosters['id']
                game_rosters['game_id'] = i
                game_rosters['game_date'] = datetime.date(
                    datetime.strptime(data['gameData']['datetime']['dateTime'][:10], "%Y-%m-%d"))

                # Append to roster master
                roster_master = roster_master.append(game_rosters)

            else:
                break

        except KeyError:
            continue

    # Subset and dedup
    roster_master_clean = roster_master.groupby(
        ['nhl_id', 'active', 'rosterStatus', 'shootsCatches', 'currentTeam', 'primaryPosition', 'height', 'weight',
         'fullName', 'birthCountry', 'birthDate']).agg({'game_date': 'max'}).reset_index()

    # Select most recent row for player
    roster_master_recent = roster_master_clean.groupby(['nhl_id']).last().rename(
        columns={"currentTeam": "team", "primaryPosition": "pos", "birthDate": "dob", "fullName": "player_name",
                 "primaryNumber": "no", "birthCountry": "pob"})
    roster_master_recent['season'] = str(season) + str(season + 1)

    roster_master_recent['dob'] = pd.to_datetime(roster_master_recent['dob'][0:9], format="%Y-%m-%d")

    roster_master_recent.reset_index(inplace=True)

    ## Prior seasons, stack and dedup
    roster_all = pd.DataFrame()

    szns = np.arange(20122013, int(str(season) + str(season + 1)) + 1, 10001)

    for szn in szns:
        filename = 'roster_master_recent_' + str(szn) + '.csv'

        roster_season = pd.read_csv(filename, encoding='utf-8')

        roster_season['season'] = szn
        roster_season = roster_season.drop(['firstName'], axis=1)
        roster_season['game_date'] = str(szn)[4:9] + '-03-31'

        roster_all = roster_all.append(roster_season)

    # Stack and dedup
    roster_all = roster_all.append(roster_master_recent, ignore_index=True)

    roster_all['dob'] = pd.to_datetime(roster_all['dob'], format="%Y-%m-%d")
    roster_all['season'] = roster_all['season'].astype(int)

    roster_all = pd.DataFrame(roster_all).reset_index(drop=True)

    ## Stack and dedup
    roster_all = roster_all.sort_values(['nhl_id', 'season']).groupby(['nhl_id', 'season']).last()

    pipeline_functions.write_boto_s3(roster_all, 'hockey-all', "roster_season_master")


def roster_info_update(season):
    """
    Update hockey_roster_info
    """
    hockey_roster_info = pipeline_functions.read_boto_s3('hockey-all', 'hockey_roster_info.csv')

    # URL for the season
    url = "https://statsapi.web.nhl.com/api/v1/teams?expand=team.roster&season=" + str(season)

    # Read All Rosters
    all_rosters = requests.get(url).json()

    rosterAll = pd.DataFrame()
    ### Loop through each team
    for i in range(len(all_rosters['teams'])):
        team_roster = all_rosters['teams'][i]['roster']['roster']

        for j in range(len(team_roster)):
            player_pd = pd.DataFrame(team_roster[j]['person'], index=[0]).drop(['link'], axis=1)

            player_pd = player_pd.rename(index=str, columns={"id": "playerId", "fullName": "playerName"})

            player_info = requests.get(
                "https://statsapi.web.nhl.com/api/v1/people/" + str(int(player_pd['playerId']))).json()

            player_info = player_info['people'][0]

            player_pd['Team'] = all_rosters['teams'][i]['abbreviation']
            player_pd['TeamName'] = all_rosters['teams'][i]['name']
            # player_pd['Pos'] = pd.Series(team_roster[j]['position']['code'])
            player_pd['playerPositionCode'] = player_info['primaryPosition']['code']
            # player_pd['Num'] = pd.Series(team_roster[j]['jerseyNumber'])
            player_pd['seasonId'] = season
            player_pd['active'] = player_info['active']
            player_pd['rosterStatus'] = player_info['rosterStatus']
            player_pd['playerBirthCountry'] = player_info['birthCountry']
            player_pd['playerBirthDate'] = player_info['birthDate']
            player_pd['playerFirstName'] = player_info['firstName']
            player_pd['playerLastName'] = player_info['lastName']
            player_pd['playerHeight'] = player_info['height']
            player_pd['playerWeight'] = player_info['weight']
            # player_pd['playerNationality'] = player_info['nationality']

            try:
                player_pd['playerBirthStateProvince'] = player_info["birthCity"]
            except:
                pass
            try:
                player_pd['birthStateProvince'] = player_info["birthStateProvince"]
            except:
                pass
            try:
                player_pd['playerShootsCatches'] = player_info['shootsCatches']
            except:
                pass
            rosterAll = rosterAll.append(player_pd)

    hockey_roster_info = hockey_roster_info.append(rosterAll).drop_duplicates()

    pipeline_functions.write_boto_s3(hockey_roster_info, 'hockey-all', 'hockey_roster_info.csv')
