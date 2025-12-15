import pandas as pd
import numpy as np
import sklearn
from Anime_Data import AnimeData

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict
from collections import Counter

import spacy
from spacy.tokens import Token

from gensim.models import Word2Vec

from multiprocessing import cpu_count

from UsefulMethods import fetch_current_anime_dict
from UsefulMethods import load
from UsefulMethods import get_genres_from_dict

_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
_nlp.max_length = 2_000_000 

def _tokens_from_texts(texts):  # NEW
    toks = []
    for doc in _nlp.pipe(texts, batch_size=128, n_process=max(1, min(4, cpu_count()-1))):
        toks.append([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])
    return toks

_VECTORIZER = None
_E = None
_TOP1K_IDS = None
_TOP1K_WORD_FEATS = None
_USER_TOK_CACHE = {} 

def _ensure_corpus_cache(top1k_dict):
    global _VECTORIZER, _E, _TOP1K_IDS, _TOP1K_WORD_FEATS
    if _VECTORIZER is not None:
        return

    _TOP1K_IDS = list(top1k_dict.keys())
    top1k_descs = [top1k_dict[i]["Synopsis"] for i in _TOP1K_IDS]

    top1k_tokens = _tokens_from_texts(top1k_descs)

    _VECTORIZER = TfidfVectorizer(
        max_features=3000,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        lowercase=False,
    ).fit(top1k_tokens)

    W2V = Word2Vec(
        sentences=top1k_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,
        workers=max(1, min(4, cpu_count()-1)),
        epochs=5,
    )

    idx2tok = np.array(_VECTORIZER.get_feature_names_out())
    V, D = len(idx2tok), W2V.vector_size
    E = np.zeros((V, D), dtype=np.float32)
    for j, tok in enumerate(idx2tok):
        if tok in W2V.wv:
            E[j] = W2V.wv[tok]
    _E = E

    top1k_tfidf = _VECTORIZER.transform(top1k_tokens)
    row_sums = np.asarray(top1k_tfidf.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    _TOP1K_WORD_FEATS = (top1k_tfidf @ _E) / row_sums[:, None]

class RidgeReg:
    __top1kanimes = load("top1kanime.json")

    def __init__(self, anime_data):
        self.__current_user_id = anime_data.get_current_user_id()
        self.__current_anime_dict = anime_data.get_current_anime_dict()
        self.__all_genres = get_genres_from_dict(RidgeReg.__top1kanimes)
        self.__train_descriptions = [synopsis["Synopsis"] for synopsis in self.__current_anime_dict.values()]
        self.__top1k_descriptions = [synopsis["Synopsis"] for synopsis in RidgeReg.__top1kanimes.values()]
        self.__all_descriptions = self.__train_descriptions + self.__top1k_descriptions

    def _get_user_tokens(self):
        key = self.__current_user_id
        cached = _USER_TOK_CACHE.get(key)
        if cached and len(cached) == len(self.__train_descriptions):
            return cached
        toks = _tokens_from_texts(self.__train_descriptions)  # n_process=1
        _USER_TOK_CACHE[key] = toks
        return toks

    def _tokens_from_texts(texts):  # NEW
        # Single pass lemmatization using spaCy pipe (fast & parallel)
        toks = []
        for doc in _nlp.pipe(texts, batch_size=128, n_process=max(1, min(4, cpu_count()-1))):
            toks.append([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])
        return toks

    def get_word_features(self):
        _ensure_corpus_cache(RidgeReg.__top1kanimes)

        train_tokens = self._get_user_tokens()

        train_tfidf = _VECTORIZER.transform(train_tokens)

        train_row_sums = np.asarray(train_tfidf.sum(axis=1)).ravel()
        train_row_sums[train_row_sums == 0] = 1.0
        train_word_features = (train_tfidf @ _E) / train_row_sums[:, None]
        train_word_features = train_word_features.astype(np.float32)

        top1k_word_features = _TOP1K_WORD_FEATS

        return train_word_features, top1k_word_features

    def get_recs(self, anime_data):
        features = []
        top1kfeatures = []
        targets = []

        train_word_features, top1k_word_features = self.get_word_features()
        for value, desc_embedding in zip(self.__current_anime_dict.values(), train_word_features):
            row = {genre: 0 for genre in self.__all_genres}
            for genre in value.get("Genres", []):
                row[genre] = 1
            row["MAL_score"] = value.get("Score")
            row["watching"] = value.get("statistics").get("status").get("watching")
            row["completed"] = value.get("statistics").get("status").get("completed")
            row["on_hold"] = value.get("statistics").get("status").get("on_hold")
            row["plan_to_watch"] = value.get("statistics").get("status").get("plan_to_watch")
            row["Rank"] = value.get("Rank")
            row["num_list_users"] = value.get("num_list_users")
            row["num_scoring_users"] = value.get("num_scoring_users")
            row["desc_embedding"] = desc_embedding
            features.append(row)

            user_rating = value.get("User_ratings", {}).get(str(anime_data.get_current_user_id()))
            targets.append(user_rating)

        X = pd.DataFrame(features)
        y = np.array(targets, dtype=float)

        embeddings = np.vstack(X["desc_embedding"].values)        # shape: (N_docs, 100)
        emb_cols = [f"desc_emb_{i}" for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=X.index)

        X = X.drop(columns=["desc_embedding"]).join(emb_df)

        X.fillna(0, inplace=True)
        mask = ~np.isnan(y)
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask]

        # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1,
        #                                                                             shuffle=True)
        alphas = np.logspace(-3, 3, 12)
        model = linear_model.RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error')
        model.fit(X, y)

        # feature_names = list(X.columns)
        #
        # # Get the coefficients from Ridge
        # coefs = model.coef_
        #
        # # Put into DataFrame
        # coef_df = pd.DataFrame({
        #     "feature": feature_names,
        #     "coef": coefs,
        #     "abs_coef": np.abs(coefs)
        # })
        #
        # # Sort by absolute importance
        # coef_df = coef_df.sort_values(by="abs_coef", ascending=False)
        #
        # print(coef_df.head(20))

        # preds = model.predict(x_test)
        # rmse = np.sqrt(mean_squared_error(y_test, preds))
        # mae = mean_absolute_error(y_test, preds)
        # baseline = np.mean(np.abs(y_test - np.mean(y_train)))
        # improvement = baseline - mae
        #
        # print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        # print(f"Baseline: {baseline}")
        # print(f"Train CV MAE: {-model.best_score_:.3f}")
        # print(f"Improvement: {improvement}")

        top1k_ids = list(RidgeReg.__top1kanimes.keys())
        top1k_descriptions = [RidgeReg.__top1kanimes[i]["Synopsis"] for i in top1k_ids]

        titles_in_order = []
        for anime_id, desc_embedding in zip(top1k_ids, top1k_word_features):
            if str(anime_id) in self.__current_anime_dict:
                continue
            value = RidgeReg.__top1kanimes[anime_id]
            row = {genre: 0 for genre in self.__all_genres}
            for genre in value.get("Genres", []):
                row[genre] = 1
            row["MAL_score"] = value.get("Score")
            row["watching"] = value.get("statistics").get("status").get("watching")
            row["completed"] = value.get("statistics").get("status").get("completed")
            row["on_hold"] = value.get("statistics").get("status").get("on_hold")
            row["plan_to_watch"] = value.get("statistics").get("status").get("plan_to_watch")
            row["Rank"] = value.get("Rank")
            row["num_list_users"] = value.get("num_list_users")
            row["num_scoring_users"] = value.get("num_scoring_users")
            row["desc_embedding"] = desc_embedding
            top1kfeatures.append(row)
            titles_in_order.append(value.get("Title", ""))

        top1kfeatures_df = pd.DataFrame(top1kfeatures)

        if top1kfeatures_df.empty:
            print("No unwatched items to recommend.")
            exit()

        embeddings_top = np.vstack(top1kfeatures_df["desc_embedding"].values)
        emb_df_top = pd.DataFrame(embeddings_top, columns=emb_cols, index=top1kfeatures_df.index)

        top1kfeatures_df = top1kfeatures_df.drop(columns=["desc_embedding"]).join(emb_df_top)

        top1kfeatures_df.fillna(0, inplace=True)
        top1kfeatures_df = top1kfeatures_df.reindex(columns=X.columns, fill_value=0.0)

        if top1kfeatures_df.empty:
            return np.array([]), []

        predictions = model.predict(top1kfeatures_df)

        return predictions, titles_in_order

    def print_recs(self, anime_data, mal_recs):
        predictions, titles_in_order = self.get_recs(anime_data)
        if predictions.size == 0:
            print("No unwatched items to recommend.")
            return

        anime_recs = defaultdict(float)

        try:
            threshold = int(input("Show recommendations predicted to be at least what rating? [default 7]: ") or 7)
        except ValueError:
            threshold = 7
        for pred, title in zip(predictions, titles_in_order):
            if pred > threshold:
                anime_recs[title] = float(pred)

        sorted_anime_recs = sorted(anime_recs.items(), key=lambda x: -x[1])

        ani_recs = [mal_recs.values()]
        ani_rec_set = set(ani_recs)
        ml_rec_set = set([anime[0] for anime in sorted_anime_recs])
        highly_rec = list(ani_rec_set & ml_rec_set)

        printed = set()
        i = 1

        if all(pred < threshold for pred in predictions):
            print("Note: None of these are high-confidence picks, but here are your top matches.")

        if not anime_recs:
            top3 = sorted(zip(predictions, titles_in_order), key=lambda x: -x[0])[:3]
            for i, (pred, anime) in enumerate(top3, 1):
                print(f"{i}: {title}")
        else:
            i = 1
            for anime in highly_rec:
                print(f"{i}: {anime} (Highly Recommended)")
                printed.add(anime)
                i += 1
            for anime_rec in sorted_anime_recs:
                if i > 3:
                    break
                if anime_rec[0] not in printed:
                    print(f"{i}: {anime_rec[0]}")
                    printed.add(anime_rec[0])
                    i += 1

    def get_recs_text(self, anime_data, mal_recs):
        predictions, titles_in_order = self.get_recs(anime_data)
        if predictions.size == 0:
            return ["No unwatched items to recommend."]

        from collections import defaultdict
        anime_recs = defaultdict(float)

        threshold = 7
        for pred, title in zip(predictions, titles_in_order):
            if pred > threshold:
                anime_recs[title] = float(pred)

        sorted_anime_recs = sorted(anime_recs.items(), key=lambda x: -x[1])

        ani_recs = [*mal_recs.values()]
        ani_rec_set = set(ani_recs)
        ml_rec_set = set([anime[0] for anime in sorted_anime_recs])
        highly_rec = list(ani_rec_set & ml_rec_set)

        printed = set()
        result_lines = []

        if all(pred < threshold for pred in predictions):
            result_lines.append("Note: None of these are high-confidence picks, but here are your top matches.")

        if not anime_recs:
            top3 = sorted(zip(predictions, titles_in_order), key=lambda x: -x[0])[:3]
            for i, (pred, anime) in enumerate(top3, 1):
                result_lines.append(f"{i}: {anime}")
        else:
            i = 1
            for anime in highly_rec:
                result_lines.append(f"{i}: {anime} (Highly Recommended)")
                printed.add(anime)
                i += 1
            for anime_rec in sorted_anime_recs:
                if i > 3:
                    break
                if anime_rec[0] not in printed:
                    result_lines.append(f"{i}: {anime_rec[0]}")
                    printed.add(anime_rec[0])
                    i += 1

        return result_lines