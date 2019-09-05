from collections import defaultdict
from functools import partial
import json

import numpy as np
import pandas as pd


def score_drives(start_season, end_season, exclude_playoffs=False,
                 exclude_blowouts=False, opponent_adjustment=True):
    """Score each drive from start_season to end_season."""
    df = get_drives(
        start_season, end_season, exclude_playoffs
    )
    if exclude_blowouts:
        df['offensive_differential'] = (
            df['offensive_team_score_start'] - df['defensive_team_score_start']
        )
        df = df.loc[
            ~((df['offensive_differential'] >= exclude_blowouts) &
              (df['start_quarter'] == 4))
        ]
    df['nfl_avg_score'] = df.groupby('start_yard_line_bin')\
        ['drive_score'].transform('mean')
    df['drive_score'] = df['drive_score'] - df['nfl_avg_score']
    if opponent_adjustment:
        df = opponent_strength_adjustment(df)
    return df


def opponent_strength_adjustment(df, n_iters=5, step_size=.2):
    # MAKE ADJUSTMENT BASED ON OPPONENT STRENGTH...
    df['adj_offensive_score'] = df['drive_score']
    df['adj_defensive_score'] = df['drive_score']
    for i in range(n_iters):
        df['offensive_adj'] = df.groupby(['season', 'defensive_team'])\
            ['adj_defensive_score'].transform('mean')
        df['defensive_adj'] = df.groupby(['season', 'offensive_team'])\
            ['adj_offensive_score'].transform('mean')
        df['adj_offensive_score'] = (
            df['adj_offensive_score'] - (step_size * df['offensive_adj'])
        )
        df['adj_defensive_score'] = (
            df['adj_defensive_score'] - (step_size * df['defensive_adj'])
        )
    df = df.drop(['offensive_adj', 'defensive_adj'], axis=1)
    return df


def get_side_columns(side):
    if side == 'offensive_team':
        columns = ['adj_offensive_score', 'drive_score']
    else:
        columns = ['adj_defensive_score', 'drive_score']
    return columns


def get_drives(start_season, end_season, exclude_playoffs):
    seasons = list(range(start_season, end_season + 1))
    df = pd.DataFrame()
    for season in seasons:
        drives = json.load(open('./data/%i_drives.json' % season, 'r'))
        sdf = preprocess_drives(drives, exclude_playoffs)
        sdf['season'] = season
        df = pd.concat((df, sdf))
    df = postprocess_drives(df)
    return df


def preprocess_drives(drives, exclude_playoffs):
    """Preprocess drives for analysis."""
    df = pd.DataFrame(drives)
    df['drive_id'] = df.index
    df['season'] = df['game_id'].map(get_season)

    if 'home_score' in df.columns:
        df['home_final_score'] = df['home_score']
        df['away_final_score'] = df['away_score']
        df =  df.drop(['home_score', 'away_score'], axis=1)

    df = clean_games(df)

    df = mark_playoffs(df)
    if exclude_playoffs:
        df = df.loc[df['is_playoffs'] == 0].copy()

    df['total_yards'] = df['penalty_yards'] + df['yards_gained']
    df['end_yard_line'] = df['start_yard_line'] + df['total_yards']
    df = get_next_opponent_drive(df)

    df = bin_yard_lines(df, binned_column='start_yard_line', prefix='start')
    df = bin_yard_lines(df, binned_column='end_yard_line', prefix='end')
    df = bin_yard_lines(df, binned_column='next_start_yard_line', prefix='next_start')

    df = mark_offensive_scores(df)
    df = mark_dst_scores(df)
    df = get_current_score(df)

    df['drive_time'] = df['drive_time'].map(convert_drive_time)
    df = format_final_scores(df)
    return df


def convert_drive_time(drive_time):
    if drive_time:
        minutes, seconds = [int(t) for t in drive_time.split(':')]
        return minutes + seconds / 60
    return None


def get_season(game_id):
    month = int(str(game_id)[4:6])
    year = int(str(game_id)[:4])
    if month > 8:
        return year
    else:
        return year - 1


def clean_games(df):
    pro_bowl_teams = ['APR', 'NPR', 'AFC', 'NFC', 'IRV', 'CRT', 'RIC', 'SAN']
    df = df.loc[~df['away_team'].isin(pro_bowl_teams)].copy()
    team_columns = [
        'away_team', 'home_team', 'offensive_team', 'defensive_team'
    ]
    team_map = {'STL': 'LA', 'SD': 'LAC', 'JAC': 'JAX'}
    for column in team_columns:
        df[column] = df[column].map(lambda team: team_map.get(team, team))
    return df


def mark_playoffs(df):
    df = df.sort_values(['game_id', 'start_quarter', 'start_time'],
                        ascending=[True, True, False])
    df['unique_game_flag'] = 0
    firstplay_mask = (df['start_quarter'] == 1) & (df['start_time'] == '15:00')
    df.loc[firstplay_mask, 'unique_game_flag'] = 1
    group_games = df.groupby('season')
    df['game_in_season'] = group_games['unique_game_flag'].transform('cumsum')
    df['is_playoffs'] = 0
    df.loc[df['game_in_season'] > 256, 'is_playoffs'] = 1
    return df


def postprocess_drives(df):
    # To get decade averages run functions here.
    df = add_field_goal_points(df)
    df = add_field_position_points(df)
    df = add_win_loss(df)
    df['offense_home'] = df['offensive_team'] == df['home_team']
    df['defense_home'] = df['defensive_team'] == df['home_team']
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


def mark_offensive_scores(df):
    """Mark tds, field goals, extra points, and two-point conversions."""
    df['expected_points'] = 0
    df['offensive_points'] = 0
    td_mask = df['result'] == 'Touchdown'
    field_goal_mask = df['result'] == 'Field Goal'
    extra_point_mask = df['last_play_desc'].str.contains('extra point is GOOD')
    two_point_mask = (
        (df['last_play_desc'].str.contains(r'TWO-POINT CONV.*SUCCEEDS')) &
        (~df['last_play_desc'].str.contains(r'SUCCEEDS.*REVERSED')) &
        (~df['last_play_desc'].str.contains(r'SUCCEEDS.*NULLIFIED'))
    )
    df.loc[td_mask, 'expected_points'] = 7
    df.loc[td_mask, 'offensive_points'] += 6
    df.loc[field_goal_mask, 'offensive_points'] += 3
    df.loc[extra_point_mask, 'offensive_points'] += 1
    df.loc[two_point_mask, 'offensive_points'] += 2
    df['is_touchdown'] = 0
    df['is_field_goal'] = 0
    df['is_score'] = 0
    df.loc[td_mask, 'is_touchdown'] = 1
    df.loc[field_goal_mask, 'is_field_goal'] = 1
    df.loc[(td_mask) | (field_goal_mask), 'is_score'] = 1
    return df


def mark_dst_scores(df):
    """Mark tds and safeties. Assume extra point made."""
    int_mask = df['result'] == 'Interception'
    fumble_mask = df['result'] == 'Fumble'
    df['is_interception'] = 0
    df.loc[int_mask, 'is_interception'] = 1
    df['is_fumble'] = 0
    df.loc[fumble_mask, 'is_fumble'] = 1
    td_mask = (
        (df['result'] != 'Touchdown') &
        (df['last_play_desc'].str.contains(r'TOUCHDOWN')) &
        (~df['last_play_desc'].str.contains(r'TOUCHDOWN.*REVERSED')) &
        (~df['last_play_desc'].str.contains(r'TOUCHDOWN.*NULLIFIED'))
    )
    safety_mask = df['result'].isin(['Safety', 'Fumble, Safety'])
    df.loc[(int_mask) & (td_mask), 'result'] = 'Interception, Touchdown'
    df.loc[(fumble_mask) & (td_mask), 'result'] = 'Fumble, Touchdown'
    df.loc[(int_mask) & (td_mask), 'expected_points'] = -7
    df.loc[(fumble_mask) & (td_mask), 'expected_points'] = -7
    df.loc[safety_mask, 'expected_points'] = -2
    df['dst_points'] = 0
    df.loc[td_mask, 'dst_points'] += 7 # Assume extra point made...
    df.loc[safety_mask, 'dst_points'] += 2
    return df


def get_current_score(df):
    """Get the current score of the drive."""
    # Initialize series.
    df['home_points'] = 0
    df['away_points'] = 0
    df['offensive_team_score_end'] = 0
    df['defensive_team_score_end'] = 0
    df['offensive_team_score_start'] = 0
    df['defensive_team_score_start'] = 0
    # Set masks.
    home_o_mask = df['home_team'] == df['offensive_team']
    away_o_mask = df['away_team'] == df['offensive_team']
    home_d_mask = df['home_team'] == df['defensive_team']
    away_d_mask = df['away_team'] == df['defensive_team']
    # Mark current drive points.
    df.loc[home_o_mask, 'home_points'] += df[home_o_mask]['offensive_points']
    df.loc[home_d_mask, 'home_points'] += df[home_d_mask]['dst_points']
    df.loc[away_o_mask, 'away_points'] += df[away_o_mask]['offensive_points']
    df.loc[away_d_mask, 'away_points'] += df[away_d_mask]['dst_points']
    # Mark score at end of current drive.
    df['home_score_end'] = (
        df.groupby('game_id')['home_points'].transform('cumsum')
    )
    df['away_score_end'] = (
        df.groupby('game_id')['away_points'].transform('cumsum')
    )
    df.loc[home_o_mask, 'offensive_team_score_end'] = (
        df[home_o_mask]['home_score_end']
    )
    df.loc[away_o_mask, 'offensive_team_score_end'] = (
        df[away_o_mask]['away_score_end']
    )
    df.loc[home_d_mask, 'defensive_team_score_end'] = (
        df[home_d_mask]['home_score_end']
    )
    df.loc[away_d_mask, 'defensive_team_score_end'] = (
        df[away_d_mask]['away_score_end']
    )
    # Mark score at start of current drive.
    df['home_score_start'] = df['home_score_end'] - df['home_points']
    df['away_score_start'] = df['away_score_end'] - df['away_points']
    df['offensive_team_score_start'] = (
        df['offensive_team_score_end'] - df['offensive_points']
    )
    df['defensive_team_score_start'] = (
        df['defensive_team_score_end'] - df['dst_points']
    )
    return df


def format_final_scores(df):
    df['offensive_final_score'] = 0
    df['defensive_final_score'] = 0
    home_o_mask = df['home_team'] == df['offensive_team']
    away_o_mask = df['away_team'] == df['offensive_team']
    home_d_mask = df['home_team'] == df['defensive_team']
    away_d_mask = df['away_team'] == df['defensive_team']
    df.loc[home_o_mask, 'offensive_final_score'] = (
        df[home_o_mask]['home_final_score']
    )
    df.loc[away_o_mask, 'offensive_final_score'] = (
        df[away_o_mask]['away_final_score']
    )
    df.loc[home_d_mask, 'defensive_final_score'] = (
        df[home_d_mask]['home_final_score']
    )
    df.loc[away_d_mask, 'defensive_final_score'] = (
        df[away_d_mask]['away_final_score']
    )
    return df


def add_field_goal_points(df):
    df['made_field_goal'] = 0
    df.loc[df['result'] == 'Field Goal', 'made_field_goal'] = 1
    field_goal_mask = df['result'].isin(['Field Goal', 'Missed FG', 'Blocked FG', 'Blocked FG, Downs'])
    field_goal_agg = df.loc[field_goal_mask].groupby('end_yard_line_bin')
    df.loc[field_goal_mask, 'expected_points'] = field_goal_agg['made_field_goal'].transform('mean') * 3
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
    nfl_agg = df.groupby('start_yard_line_bin')['expected_points'].mean()
    nfl_agg = nfl_agg.to_dict()
    df['expected_points_opp_from_start'] = df['start_opp_expected_yard_line_bin'].map(nfl_agg)
    df['expected_points_opp_from_end'] = df['end_opp_expected_yard_line_bin'].map(nfl_agg)
    df['field_position_points'] = (
        df['expected_points_opp_from_start'] - df['expected_points_opp_from_end']
    )
    df['drive_score'] = df['expected_points'] + df['field_position_points']
    drop_columns = [
        'start_opp_expected_start',
        'end_opp_expected_start'
    ]
    df = df.drop(drop_columns, axis=1)
    return df


def add_win_loss(df):
    df['offensive_win'] = (
        (df['offensive_team'] == df['home_team']).astype(int) *
        (df['home_final_score'] > df['away_final_score']).astype(int) +
        (df['offensive_team'] == df['away_team']).astype(int) *
       ( df['away_final_score'] > df['home_final_score']).astype(int)
    )
    df['defensive_win'] = (
        (df['defensive_team'] == df['home_team']).astype(int) *
        (df['home_final_score'] > df['away_final_score']).astype(int) +
        (df['defensive_team'] == df['away_team']).astype(int) *
       ( df['away_final_score'] > df['home_final_score']).astype(int)
    )
    df['tie'] = (df['offensive_win'] + df['defensive_win'] == 0).astype(int)
    return df
