from datetime import datetime, timedelta
import json
import pickle
import requests

import numpy as np


def season_games_pipeline(season, url, months=[9, 10, 11, 12, 1, 2]):
    """Extract all high-level game data for a season.

    Arguments:
        season (int) - year corresponding to start of season
        url (str) - API URL to pull data from

    Returns:
        season_data (dict)
    """
    season_data = get_write_games(season, url, months)
    return season_data


def format_season_start_end(season):
    # Format the start and end date of the season in question.
    start = '{}0901'.format(season)
    end = '{}0220'.format(season + 1)
    return start, end


def get_write_games(season, url, months):
    """Get all of the game ids between start and end.

    Arguments:
        season (int) - year corresponding to start of season
        url (str) - API URL to pull data from
    """
    start, end = format_season_start_end(season)
    game_date = datetime.strptime(start, '%Y%m%d')
    while game_date <= datetime.strptime(end, '%Y%m%d'):
        in_day_id = 0
        failed = False
        fail_count = 0
        game_date_str = game_date.strftime('%Y%m%d')
        while not failed and game_date.month in months:
            try:
                game_id = format_game_id(game_date_str, in_day_id)
                game = get_game(game_id, url)
                json.dump(
                    game, open('data/%i/%s.json' % (season, game_id), 'w')
                )
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


def get_game(game_id, url):
    # Request game data from NFL Game Center API.
    game = requests.get(
        '{url}{game_id}/{game_id}_gtd.json'.format(url=url, game_id=game_id)
    )
    if game.reason == 'Not Found':
        raise Exception('Game ID {} not found.'.format(game_id))
    game_dict = json.loads(game.text)
    return game_dict


if __name__ == '__main__':
    # months = [9, 10, 11, 12, 1, 2]
    months = [9]
    for season in range(2019, 2020):
        season_games_pipeline(
            season, 'http://www.nfl.com/liveupdate/game-center/', months
        )
