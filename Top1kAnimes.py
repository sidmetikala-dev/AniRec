import requests
import json
import os
import time
from UsefulMethods import fetch_anime_details

client_id = '221713b4eb66c9198bdcf0825f746a11'

url = 'https://api.myanimelist.net/v2/anime/ranking'

headers = {
    "X-MAL-CLIENT-ID": client_id
}

params = {
    "ranking_type": "all",
    "limit": 500,
    "offset": 500
}

cache_file = "top1kanime.json"
if os.path.exists(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        topAnime = json.load(f)
else:
    topAnime = {}

response = requests.get(url, headers=headers, params=params)

data = response.json()

top1kAnime_ids = []
for anime in data['data']:
    top1kAnime_ids.append(anime['node']['id'])

for anime_id in top1kAnime_ids:
    if str(anime_id) in topAnime:
        continue
    anime_data = fetch_anime_details(anime_id, client_id)
    if anime_data:
        if anime_data['media_type'] == 'tv':
            # Populate the dictionary with anime data
            topAnime[anime_data['id']] = {
                "Title": anime_data.get('title', ''),
                "Synopsis": anime_data.get('synopsis', ''),
                "Score": anime_data.get('mean', ''),
                "Rank": anime_data.get('rank', ''),
                "num_list_users": anime_data.get('num_list_users', ''),
                "num_scoring_users": anime_data.get('num_scoring_users', ''),
                "media_type": anime_data.get('media_type', ''),
                "Status": anime_data.get('status', ''),
                "Genres": [genre.get('name', '') for genre in anime_data.get('genres', [])],
                "Genre_Ids": [genre.get('id', '') for genre in anime_data.get('genres', [])],
                "num_episodes": anime_data.get('num_episodes', ''),
                "rating": anime_data.get('rating', ''),
                "studios": [studio.get('name', '') for studio in anime_data.get('studios', [])],
                "studio_ids": [studio.get('id', '') for studio in anime_data.get('studios', [])],
                "statistics": anime_data.get('statistics', ''),
                "recommendations": anime_data.get('recommendations', '')
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(topAnime, f, indent=2, ensure_ascii=False)
            time.sleep(0.5)