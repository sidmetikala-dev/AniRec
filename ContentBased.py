from AniRec import current_anime_dict
from UsefulMethods import load
from User_Data import scores
import json, os
from collections import defaultdict

top1kanimes = load("top1kanime.json")

genre_weights = defaultdict(float)
studio_weights = defaultdict(float)

for anime_id_str, anime_data in current_anime_dict.items():
    user_rating = scores.get(int(anime_id_str), 0)
    # if user_rating < 7:
    #     continue

    genres = anime_data.get("Genres", [])
    for genre in genres:
        genre_weights[genre] += user_rating - 5

    studios = anime_data.get("studios", [])
    for studio in studios:
        studio_weights[studio] += user_rating - 5

# for genre, weight in sorted(genre_weights.items(), key=lambda x: -x[1]):
#     print(f"{genre}: {weight}")

total_genre_weight = sum(abs(w) for w in genre_weights.values()) or 1
for genre in genre_weights:
    genre_weights[genre] /= total_genre_weight

total_studio_weight = sum(abs(w) for w in studio_weights.values()) or 1
for studio in studio_weights:
    studio_weights[studio] /= total_studio_weight

rec_scores = defaultdict(float)

for key, value in top1kanimes.items():
    if key in current_anime_dict:
        continue

    topAnime_genres = value.get("Genres")
    topAnime_studios = value.get("studios")

    if not topAnime_genres:
        continue

    for topGenre in topAnime_genres:
        rec_scores[value.get("Title")] += float(genre_weights.get(topGenre, 0))

    for topStudio in topAnime_studios:
        rec_scores[value.get("Title")] += float(studio_weights.get(topStudio, 0))

# for title, number in sorted(rec_scores.items(), key=lambda x: -x[1])[:3]:
#     if number > 0:
#         print(f"{title}: {number}")

# print([*studio_weights.keys()])