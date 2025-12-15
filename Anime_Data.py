from User_Data import UserData
import requests
import time
import json
import os
from User_ID import get_user_id
from UsefulMethods import fetch_current_anime_dict

class AnimeData:
    __anime_dict = {}

    def __init__(self, user_data):
        self.__anime_ids = user_data.get_anime_ids()
        self.__client_id = user_data.get_client_id()
        self.__username = user_data.get_username()
        self.__scores = user_data.get_scores()
        self.__current_anime_dict = {}
        self.__headers = {
            "X-MAL-CLIENT-ID": self.__client_id
        }

        self.__params = {
            "fields": "id,title,synopsis,mean,rank,num_list_users,num_scoring_users,media_type,status,genres,num_episodes,rating,studios,statistics,recommendations"
        }

        self.__current_user_id = get_user_id(self.__username)


    def update_anime_dict(self):
        cache_file = "anime_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                AnimeData.__anime_dict = json.load(f)
        else:
            AnimeData.__anime_dict = {}

        for anime_id in self.__anime_ids:

            anime_id_str = str(anime_id)
            user_rating = self.__scores.get(anime_id, None)

            if anime_id_str in AnimeData.__anime_dict:
                # Already cached: update user lists/ratings if needed
                entry = AnimeData.__anime_dict[anime_id_str]
                # Update User_IDs
                if "User_IDs" not in entry:
                    entry["User_IDs"] = []
                if self.__current_user_id not in entry["User_IDs"]:
                    entry["User_IDs"].append(self.__current_user_id)
                # Update User_ratings
                if "User_ratings" not in entry:
                    entry["User_ratings"] = {}
                entry["User_ratings"][str(self.__current_user_id)] = user_rating
                continue

            url = f"https://api.myanimelist.net/v2/anime/{anime_id}"

            success = False
            retries = 3
            wait = 1

            for attempt in range(retries):
                response = requests.get(url, headers=self.__headers, params=self.__params)

                if response.status_code == 200:
                    anime_data = response.json()

                    if anime_data['media_type'] == 'tv':

                        # Populate the dictionary with anime data
                        AnimeData.__anime_dict[anime_id_str] = {
                            "User_IDs": [self.__current_user_id],
                            "User_ratings": {str(self.__current_user_id): user_rating},
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
                            json.dump(AnimeData.__anime_dict, f, indent=2, ensure_ascii=False)
                    success = True
                    break
                elif response.status_code in [504, 429]:
                    print(
                        f"Retry {attempt + 1}/{retries} for anime ID {anime_id} (Error {response.status_code}). Waiting {wait}s...")
                    time.sleep(wait)
                    wait *= 2
                else:
                    print(f"Error {response.status_code}: {response.text}")
            if not success:
                print(f"Failed to fetch data for anime ID {anime_id} after {retries} attempts.")

            time.sleep(0.5)

    def get_anime_dict(self):
        return AnimeData.__anime_dict.copy()

    def get_current_anime_dict(self):
        return fetch_current_anime_dict(AnimeData.__anime_dict, self.__current_user_id)

    def get_current_user_id(self):
        return self.__current_user_id