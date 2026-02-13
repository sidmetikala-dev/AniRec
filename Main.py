from flask import Flask, jsonify, request, send_from_directory
from User_Data import UserData
from Anime_Data import AnimeData
from AniRec import AniRec
from RidgeReg_WordEmb import RidgeReg
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__)

CLIENT_ID = os.getenv("CLIENT_ID")

@app.route("/")
def root():
    return send_from_directory(".", "index.html")


@app.route("/index.js")
def js():
    return send_from_directory(".", "index.js")

@app.post("/api/recs")
def recs():

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"error": "username is required"}), 400

    user = UserData(CLIENT_ID, username)
    anime_data = AnimeData(user)
    anime_data.update_anime_dict()

    mal = AniRec(user, anime_data)
    mal_recs = mal.get_recs()

    wordemb = RidgeReg(anime_data)
    recs_text = wordemb.get_recs_text(anime_data, mal_recs)
    return jsonify(recs_text)

if __name__ == "__main__":
    app.run(debug=True)
