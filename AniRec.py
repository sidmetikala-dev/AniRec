from Anime_Data import AnimeData
from User_Data import UserData
import json
from UsefulMethods import fetch_current_anime_dict

class AniRec:

    def __init__(self, user_data, anime_data):
        self.__user_anime_ids = set(user_data.get_anime_ids())
        self.__scores = user_data.get_scores()
        self.__current_anime_dict = anime_data.get_current_anime_dict()
        self.__anime_rec = {}
        self.__count = {}


    def get_recs(self, top_k: int=3):
        for k, v in self.__current_anime_dict.items():
            # Only look at anime the current user has watched
            if int(k) not in self.__user_anime_ids or self.__scores.get(int(k), 0) < 7:
                continue
            recommendations = v.get("recommendations")
            if not recommendations:
                continue
            for node in recommendations:
                rec_anime_id = node['node']['id']
                self.__count[rec_anime_id] = self.__count.get(rec_anime_id, 0) + 1
                self.__anime_rec[rec_anime_id] = node['node']['title']

        if not self.__count:
            return {}

        ranked = sorted(self.__count.items(), key=lambda kv: (-kv[1], self.__anime_rec.get(kv[0], "")))

        # take top_k
        top = ranked[:top_k]
        return {aid: self.__anime_rec[aid] for aid, _ in top}

    def print_recs(self, recs=None, top_k: int = 3):
        if recs is None:
            recs = self.get_recs(top_k=top_k)

        if recs:
            counts = [self.__count[aid] for aid in recs]
            low_conf = all(c == 1 for c in counts)

            for anime_id in recs:
                print(recs[anime_id])

            if low_conf:
                print("\nNote: These are low-confidence recs.")
        else:
            print("Warning! We couldn't recommend anything because you don't have enough data. "
                  "We suggest rating more anime for better results.")
