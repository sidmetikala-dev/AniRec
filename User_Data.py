import requests

class UserData:

    def __init__(self, client_id, username):
        # API ACCESS
        self.__client_id = client_id
        self.__username = username

        self.__url = f"https://api.myanimelist.net/v2/users/{username}/animelist"
        self.__headers = {
            "X-MAL-CLIENT-ID": client_id
        }
        self.__params = {
            "sort": "list_score",
            "limit": 1000,
            "fields": "list_status{score}, id"}

        self.__scores = {}
        self.__response = requests.get(self.__url, headers=self.__headers, params=self.__params, timeout=15)

    def refresh(self):
        self.__response = requests.get(self.__url, headers=self.__headers, params=self.__params, timeout=15)

    def get_anime_ids(self):
        #Fetch Unique IDs for Adding Anime Desc
        self.__scores = {}

        self.refresh()

        if self.__response.status_code == 200:
            user_data = self.__response.json()

            for anime in user_data.get('data', []):
                node = anime.get('node', {})
                if not node:
                    continue
                score = anime.get("list_status", {}).get("score", 0)
                if score:
                    self.__scores[node['id']] = score

            next_url = user_data.get("paging", {}).get("next")
            while next_url:
                r = requests.get(next_url, headers=self.__headers, timeout=15)
                if r.status_code != 200:
                    print(f"Error {r.status_code}: {r.text}")
                    break
                page = r.json()
                for anime in page.get("data", []):
                    node = anime["node"]
                    score = anime.get("list_status", {}).get("score", 0)
                    if score:
                        self.__scores[node["id"]] = score
                next_url = page.get("paging", {}).get("next")

        else:
            print(f"Error {self.__response.status_code}: {self.__response.text}")

        anime_ids = [*self.__scores.keys()]

        return anime_ids

    def get_scores(self):
        return self.__scores.copy()

    def get_client_id(self):
        return self.__client_id

    def get_username(self):
        return self.__username