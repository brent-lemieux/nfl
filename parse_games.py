from datetime import datetime, timedelta
import json
import os

import numpy as np
import pandas as pd


def game_data_pipeline(start_season, end_season, data_path):
    """Get game data."""
    df = pd.DataFrame()
    for season in range(start_season, end_season + 1):
        season_data = parse_season_games(season, data_path)
        sdf = pd.DataFrame(season_data)
        df = pd.concat((df, sdf))
    return df


def parse_season_games(season, data_path):
    """Extract all high-level game data for a season.

    Arguments:
        season (int) - year corresponding to start of season
        data_path (str) - API URL to pull data from

    Returns:
        season_data (dict)
    """
    season_game_files = get_season_game_list(season, data_path)
    parsed_season_data = []
    for filename in season_game_files:
        try:
            game_path = '{}/{}/{}'.format(data_path, season, filename)
            game_data = json.load(open(game_path, 'r'))
            game_id = filename.replace('.json', '')
            parsed_game_data = parse_game(game_data, game_id)
            parsed_season_data.extend(parsed_game_data)
        except Exception as e:
            print(filename, e)
    return parsed_season_data


def get_season_game_list(season, data_path):
    season_path = '{}/{}'.format(data_path, season)
    game_list = os.listdir(season_path)
    return game_list


def parse_game(game_data, game_id):
    # Parse game into drive level data -- TODO: play by play data.
    game = game_data[game_id]
    home = game['home']['abbr']
    away = game['away']['abbr']
    drives = game['drives']
    game_drives = []
    for i in range(1, 1 + drives['crntdrv']):
        drive = drives.get(str(i), None)
        if not drive:
            continue
        offensive_team = drive['posteam']
        defensive_team = home if home != drive['posteam'] else away
        try:
            first_play_key = str(
                min([int(key) for key in drive['plays'].keys()])
            )
            last_play_key = str(
                max([int(key) for key in drive['plays'].keys()])
            )
        except Exception as e:
            print(e, drive['plays'].keys())
            continue
        drive_dict = dict(
            game_id=game_id,
            offensive_team=offensive_team,
            defensive_team=defensive_team,
            home_team=home,
            away_team=away,
            start_quarter=drive['start']['qtr'],
            start_time=drive['start']['time'],
            start_yard_line=format_yardline(drive['start'], offensive_team),
            yards_gained=drive['ydsgained'],
            penalty_yards=drive['penyds'],
            end_quarter=drive['end']['qtr'],
            end_time=drive['end']['time'],
            result=drive['result'],
            n_plays=drive['numplays'],
            drive_time=drive['postime'],
            first_play_desc=drive['plays'][first_play_key]['desc'],
            last_play_desc=drive['plays'][last_play_key]['desc'],
            home_final_score=game['home']['score']['T'],
            away_final_score=game['away']['score']['T'],
            home_score_diff_last_quarter=format_score_differential(game, drive)
        )
        game_drives.append(drive_dict)
    return game_drives


def format_yardline(start, team):
    # Format the yard line as an integer from 0 to 100.
    yard_line_str = start['yrdln']
    if yard_line_str == '50':
        return 50
    elif yard_line_str:
        side_of_field, yard_line_str = yard_line_str.split(' ')
        if side_of_field == team:
            return int(yard_line_str)
        else:
            return 100 - int(yard_line_str)
    else:
        return np.nan


def format_score_differential(game, drive):
    # Get score differential (home - away) at end of previous quarter.
    start_quarter = drive['start']['qtr']
    # print(start_quarter, type(start_quarter))
    if str(start_quarter) == '1':
        return 0
    quarters = range(1, start_quarter)
    home_scores = [game['home']['score'][str(quarter)] for quarter in quarters]
    away_scores = [game['away']['score'][str(quarter)] for quarter in quarters]
    return str(np.sum(home_scores) - np.sum(away_scores))
