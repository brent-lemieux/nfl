from datetime import datetime, timedelta
import json
import pickle
import requests


def extract_season(season, url='http://www.nfl.com/liveupdate/game-center/'):
    """Extract all high-level game data for a season.

    Arguments:
        season (int) - year corresponding to start of season
        url (str) - API URL to pull data from

    Returns:
        season_data (dict) - key = game_id, value = sub_dict; sub_dict keys =
            completed_game, home_team, home_score, away_team, away_score
    """
    start, end = format_season_start_end(season)
    season_data = get_games_data(start, end, url)
    return season_data


def format_season_start_end(season):
    # Format the start and end date of the season in question.
    start = '{}0901'.format(season)
    end = '{}0228'.format(season + 1)
    return start, end


def get_games_data(start, end, url):
    """Get all of the game ids between start and end.

    Arguments:
        start (str) - string representation of date; 'YYYYMMDD';
            get game ids with a game id date >= start
        end (str) - string representation of date; 'YYYYMMDD';
            get game ids with a game id date <= end
        url (str) - API URL to pull data from

    Returns:
        games_data (dict) - key = game_id, value = sub_dict; sub_dict keys =
            completed_game, home_team, home_score, away_team, away_score
    """
    games_data = dict()
    game_date = datetime.strptime(start, '%Y%m%d')
    while game_date <= datetime.strptime(end, '%Y%m%d'):
        in_day_id = 0
        failed = False
        game_date_str = game_date.strftime('%Y%m%d')
        while not failed and game_date.month in [9, 10, 11, 12, 1, 2]:
            game_id = format_game_id(game_date_str, in_day_id)
            game = requests.get(
                '{url}{game_id}/{game_id}_gtd.json'.format(
                    url=url, game_id=game_id
                )
            )
            if game.reason == 'Not Found':
                failed = True
            else:
                data = json.loads(game.text)
                try:
                    games_data[game_id] = get_game_summary(data[game_id])
                except Exception as e:
                    print(game_id)
            if game_date.weekday() == 6 and in_day_id <= 16:
                failed = False
            if game_date.weekday() in [0, 2, 3, 5] and in_day_id <= 5:
                failed = False
            in_day_id += 1
        game_date += timedelta(days=1)
    return games_data


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


def get_game_summary(data):
    completed_game = data['qtr'] == 'Final'
    game_data = dict(
        completed_game=completed_game,
        home_team=data['home']['abbr'],
        home_score=data['home']['score'],
        away_team=data['away']['abbr'],
        away_score=data['away']['score']
    )
    return game_data


if __name__ == '__main__':
    for i in range(2010, 2018):
        games = extract_season(i)
        json.dump(games, open('data/%i_games_dict.json' % i, 'w'))
