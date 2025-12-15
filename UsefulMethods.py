import requests
import time
import json, os


def fetch_anime_details(anime_id, client_id):
    headers = {"X-MAL-CLIENT-ID": client_id}
    params = {
        "fields": "id,title,synopsis,mean,rank,num_list_users,num_scoring_users,media_type,status,genres,num_episodes,rating,studios,statistics,recommendations"
    }
    url = f"https://api.myanimelist.net/v2/anime/{anime_id}"

    retries = 3
    wait = 1

    for attempt in range(retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code in [504, 429]:
            time.sleep(wait)
            wait *= 2
        else:
            print(f"Error {response.status_code} for anime {anime_id}: {response.text}")
            break

    return None

def fetch_current_anime_dict(anime_dict, current_user_id):
    current_anime_dict = {
        key: value
        for key, value in anime_dict.items()
        if "User_IDs" in value and current_user_id in value["User_IDs"]
    }

    return current_anime_dict

def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

def get_genres_from_dict(anime_dict):
    seen_genres = set()
    genres = [genre for value in anime_dict.values() for genre in value.get("Genres") if
              genre not in seen_genres and not seen_genres.add(genre)]
    return genres

def get_studios_from_dict(anime_dict):
    seen_studios = set()
    studios = [studio for value in anime_dict.values() for studio in value.get("studios") if
              studio not in seen_studios and not seen_studios.add(studio)]
    return studios