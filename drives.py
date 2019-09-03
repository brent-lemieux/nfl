from datetime import datetime, timedelta
import json
import pickle
import requests

import numpy as np


def season_drives_pipeline(season, url):
    """Extract all high-level game data for a season.

    Arguments:
        season (int) - year corresponding to start of season
        url (str) - API URL to pull data from

    Returns:
        season_data (dict)
    """
    start, end = format_season_start_end(season)
    season_data = get_drives(start, end, url)
    return season_data


def format_season_start_end(season):
    # Format the start and end date of the season in question.
    start = '{}0901'.format(season)
    end = '{}0220'.format(season + 1)
    return start, end


def get_drives(start, end, url):
    """Get all of the game ids between start and end.

    Arguments:
        start (str) - string representation of date; 'YYYYMMDD';
            get game ids with a game id date >= start
        end (str) - string representation of date; 'YYYYMMDD';
            get game ids with a game id date <= end
        url (str) - API URL to pull data from

    Returns:
        season_drives (dict)
    """
    season_drives = []
    game_date = datetime.strptime(start, '%Y%m%d')
    while game_date <= datetime.strptime(end, '%Y%m%d'):
        in_day_id = 0
        failed = False
        fail_count = 0
        game_date_str = game_date.strftime('%Y%m%d')
        while not failed and game_date.month in [9, 10, 11, 12, 1, 2]:
            try:
                game_id = format_game_id(game_date_str, in_day_id)
                game_drives = get_game_drives(game_id, url)
                season_drives.extend(game_drives)
                in_day_id += 1
            except Exception as e:
                print(e, format_game_id(game_date_str, in_day_id))
                if game_date.weekday()==6 and in_day_id<=16 and fail_count<5:
                    fail_count += 1
                    in_day_id += 1
                elif fail_count < 5:
                    fail_count += 1
                    in_day_id += 1
                else:
                    failed = True
        game_date += timedelta(days=1)
    return season_drives


def format_game_id(game_date_str, in_day_id):
    """Format the game_id to match the game date plus in-day id convention.

    Arguments:
        game_date_str (str) - game date; 'YYYYMMDD'
        in_day_id (int) - in-day id; a unique identifier for each game played
            on a particular day
    """
    if in_day_id > 9:
        game_date_id = str(in_day_id)
    else:
        game_date_id = '0{}'.format(in_day_id)
    return game_date_str + game_date_id


def get_game_drives(game_id, url):
    # Request game data from NFL Game Center API.
    game = requests.get(
        '{url}{game_id}/{game_id}_gtd.json'.format(url=url, game_id=game_id)
    )
    if game.reason == 'Not Found':
        raise Exception('Game ID {} not found.'.format(game_id))
    game_dict = json.loads(game.text)
    return parse_game_drives(game_dict, game_id)


def parse_game_drives(game_dict, game_id):
    # Parse game drive data.
    game = game_dict[game_id]
    home = game['home']['abbr']
    away = game['away']['abbr']
    drives = game['drives']
    team_game_drives = []
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
        team_game_drives.append(drive_dict)
    return team_game_drives


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





if __name__ == '__main__':
    seasons = [2018]
    for season in seasons:
        drives = season_drives_pipeline(
            season, 'http://www.nfl.com/liveupdate/game-center/'
        )
        json.dump(drives, open('data/%i_drives.json' % season, 'w'))

    # for season in range(2009, 2019):
    #     drives = season_drives_pipeline(
    #         season, 'http://www.nfl.com/liveupdate/game-center/'
    #     )
    #     json.dump(drives, open('data/%i_drives.json' % season, 'w'))
