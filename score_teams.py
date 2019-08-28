from collections import defaultdict
from functools import partial
import json

import numpy as np
import pandas as pd


def score_drives(start_season, end_season):
    """Score each drive from start_season to end_season."""
    df = get_drives(start_season, end_season)
    df = postprocess_drives(df)
    df['nfl_avg_score'] = df.groupby('start_yard_line_bin')\
        ['drive_score'].transform('mean')
    df['drive_score'] = df['drive_score'] - df['nfl_avg_score']
    return df


def aggregate_game_drives(start_season, end_season, side='offensive_team',
                          opponent_strength=False):
    """Score each game by aggregating drive scores by game_id and team.

    Adjust for opponent strength if opponent_strength == True.
    """
    df = score_drives(start_season, end_season)
    other = 'defensive_team' if side == 'offensive_team' else 'offensive_team'
    gdf = df.groupby(['game_id', side, other, 'season'], as_index=False)['drive_score'].mean()
    gdf = gdf.sort_values('drive_score', ascending=side == 'defensive_team')
    if opponent_strength:
        gdf = opponent_strength_adjustment(
            gdf, start_season, end_season, side, other
        )
    return gdf


def aggregate_season_drives(start_season, end_season, side='offensive_team',
                            opponent_strength=False):
    """Score each season by aggregating game drive scores by season and team.

    Adjust for opponent strength if opponent_strength == True.
    """
    gdf = aggregate_game_drives(start_season, end_season, side,
                                opponent_strength)
    if opponent_strength:
        sdf = gdf.groupby(['season', side], as_index=False)[['drive_score', 'adj_drive_score']].mean()
        sdf = sdf.sort_values('adj_drive_score', ascending=False)
    else:
        sdf = gdf.groupby(['season', side], as_index=False)['drive_score'].mean()
        sdf = sdf.sort_values('drive_score')
    return sdf



def opponent_strength_adjustment(gdf, start_season, end_season, side, other):
    # MAKE ADJUSTMENT BASED ON OPPONENT STRENGTH...
    opponent_dict = defaultdict(dict)
    opponent_df = aggregate_season_drives(start_season, end_season, side=other)
    for season, team, score in opponent_df.values:
        opponent_dict[season][team] = score
    def get_opponent_score(row, opponent_dict):
        # Get the defenses season average.
        return opponent_dict[row['season']][row[other]]
    opponent_score_func = partial(
        get_opponent_score, opponent_dict=opponent_dict
    )
    gdf['opponent_adjustment'] = gdf.apply(opponent_score_func, axis=1)
    gdf['adj_drive_score'] = gdf['drive_score'] - gdf['opponent_adjustment']
    return gdf


def get_drives(start_season, end_season):
    teams = [
        'PHI', 'ATL', 'BUF', 'BAL', 'CLE', 'PIT', 'IND', 'CIN', 'MIA',
        'TEN', 'SF', 'MIN', 'HOU', 'NE', 'TB', 'NO', 'NYG', 'JAX', 'KC',
        'LAC', 'ARI', 'WAS', 'CAR', 'DAL', 'SEA', 'DEN', 'CHI', 'GB',
        'DET', 'NYJ', 'LA', 'OAK', 'JAC', 'SD', 'STL'
    ]
    seasons = list(range(start_season, end_season + 1))
    df = pd.DataFrame()
    for season in seasons:
        drives = json.load(open('./data/%i_drives.json' % season, 'r'))
        sdf = preprocess_drives(drives)
        sdf['season'] = season
        df = pd.concat((df, sdf))
    return df


def preprocess_drives(drives):
    """Preprocess drives for analysis."""
    df = pd.DataFrame(drives)
    df['drive_id'] = df.index
    df = df.loc[~df['away_team'].isin(['APR', 'NPR', 'AFC', 'NFC', 'IRV', 'CRT'])].copy()

    df['total_yards'] = df['penalty_yards'] + df['yards_gained']
    df['end_yard_line'] = df['start_yard_line'] + df['total_yards']
    df = get_next_opponent_drive(df)

    df = bin_yard_lines(df, binned_column='start_yard_line', prefix='start')
    df = bin_yard_lines(df, binned_column='end_yard_line', prefix='end')
    df = bin_yard_lines(df, binned_column='next_start_yard_line', prefix='next_start')

    df = add_offensive_scores(df)
    df = subtract_defensive_scores(df)
    return df


def postprocess_drives(df):
    # To get decade averages run functions here.
    df = add_field_goal_points(df)
    df = add_field_position_points(df)
    return df


def get_next_opponent_drive(df):
    # Get the opponents next drive.
    df['next_start_yard_line'] = df['start_yard_line'].shift(-1)
    df['next_end_yard_line'] = df['end_yard_line'].shift(-1)
    df['next_offensive_team'] = df['offensive_team'].shift(-1)
    df['next_home_team'] = df['home_team'].shift(-1)
    df['next_away_team'] = df['away_team'].shift(-1)
    same_team_mask = df['offensive_team'] == df['next_offensive_team']
    new_game_mask = (df['home_team'] != df['next_home_team']) | (df['away_team'] != df['next_away_team'])
    df.loc[(same_team_mask) | (new_game_mask), 'next_start_yard_line'] = np.nan
    df.loc[(same_team_mask) | (new_game_mask), 'next_end_yard_line'] = np.nan
    df.loc[new_game_mask, 'next_offensive_team'] = np.nan
    df = df.drop(['next_home_team', 'next_away_team'], axis=1)
    return df


def bin_yard_lines(df, binned_column, prefix):
    # Bin yard lines into groups of five.
    lower = np.arange(0, 100, 10)
    upper = np.arange(10, 110, 10)
    bins_list = list(zip(lower, upper))
    bins = pd.IntervalIndex.from_tuples(bins_list)
    df['%s_yard_line_bin' % prefix] = pd.cut(df[binned_column], bins)
    df['%s_yard_line_bin' % prefix] = df['%s_yard_line_bin' % prefix].map(
        lambda x: '%s-%s' % (x.left, x.right)
    )
    return df


def add_offensive_scores(df):
    df['points'] = 0
    df.loc[df['result'] == 'Touchdown', 'points'] = 7
    return df


def subtract_defensive_scores(df):
    # Alter result of fumble and interceptions that result in defensive TD.
    int_mask = df['result'] == 'Interception'
    fumble_mask = df['result'] == 'Fumble'
    td_mask = df['last_play_desc'].str.contains('TOUCHDOWN')
    safety_mask = df['result'].isin(['Safety', 'Fumble, Safety'])
    df.loc[(int_mask) & (td_mask), 'result'] = 'Interception, Touchdown'
    df.loc[(fumble_mask) & (td_mask), 'result'] = 'Fumble, Touchdown'
    df.loc[(int_mask) & (td_mask), 'points'] = -7
    df.loc[(fumble_mask) & (td_mask), 'points'] = -7
    df.loc[safety_mask, 'points'] = -2
    return df


def add_field_goal_points(df):
    df['made_field_goal'] = 0
    df.loc[df['result'] == 'Field Goal', 'made_field_goal'] = 1
    field_goal_mask = df['result'].isin(['Field Goal', 'Missed FG', 'Blocked FG', 'Blocked FG, Downs'])
    field_goal_agg = df.loc[field_goal_mask].groupby('end_yard_line_bin')
    df.loc[field_goal_mask, 'points'] = field_goal_agg['made_field_goal'].transform('mean') * 3
    df = df.drop('made_field_goal', axis=1)
    return df


def add_field_position_points(df):
    """Add or subtract points based on field position changes.

    1. Where does the average team get the ball based on your start_yard_line?
    2. How many expected points is that worth?
    3. Where does the average team get the ball based on your end_yard_line?
    4. How many expected points is that worth?
    5. What is the change in your opponents expected points on their next drive?
    """
    df['start_opp_expected_start'] = df.groupby('start_yard_line_bin')\
        ['next_start_yard_line'].transform('mean')
    df['end_opp_expected_start'] = df.groupby('end_yard_line_bin')\
        ['next_start_yard_line'].transform('mean')
    df = bin_yard_lines(
        df, binned_column='start_opp_expected_start', prefix='start_opp_expected'
    )
    df = bin_yard_lines(
        df, binned_column='end_opp_expected_start', prefix='end_opp_expected'
    )
    df['expected_points'] = df['points']
    nfl_agg = df.groupby('start_yard_line_bin')['points'].mean()
    nfl_agg = nfl_agg.to_dict()
    df['expected_points_opp_from_start'] = df['start_opp_expected_yard_line_bin'].map(nfl_agg)
    df['expected_points_opp_from_end'] = df['end_opp_expected_yard_line_bin'].map(nfl_agg)
    df['field_position_points'] = (
        df['expected_points_opp_from_start'] - df['expected_points_opp_from_end']
    )
    df['drive_score'] = df['points'] + df['field_position_points']
    drop_columns = [
        'expected_points',
        'start_opp_expected_start',
        'end_opp_expected_start'
    ]
    df = df.drop(drop_columns, axis=1)
    return df
