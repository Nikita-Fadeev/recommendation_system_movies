import os
import uuid
import polars as pl

import requests
from flask import Flask, render_template, session, request

app = Flask(__name__)
app.secret_key = os.urandom(24)

HOST = 'http://157.180.37.79'
recommendation_service_url = f"{HOST}:5001"
interactions_url = f"{HOST}:5000"

current_dir = os.path.dirname(os.path.abspath(__file__))
links_path = os.path.join(current_dir, 'static', 'links.csv')
movies_path = os.path.join(current_dir, 'static', 'movies.csv')

links_data = (
    pl.read_csv(links_path)
    .with_columns(pl.col('movieId').cast(pl.Utf8))
)
movies_data = (
    pl.read_csv(movies_path)
    .with_columns(pl.col('movieId').cast(pl.Utf8))
)


def imdb_url(imdb_id):
    imdb_id = str(imdb_id)
    return 'https://www.imdb.com/title/tt' + '0' * (7 - len(imdb_id)) + imdb_id


movie_id_title = {
    movie_id: title
    for movie_id, title in movies_data.select('movieId', 'title').rows()
}
movie_id_imdb = {
    movie_id: imdb_url(imdb_id)
    for movie_id, imdb_id in links_data.select('movieId', 'imdbId').rows()
}
# отображаем только топ-12 рекомендаций
TOP_K = 12


@app.route('/')
def index():
    # если передан идентификатор пользователя, используем его
    user_id = request.args.get('user_id', None)

    if user_id is None:
        # пробуем достать user_id из сессии, если это не холодный пользователь
        user_id = session.get('user_id')

        # если пользователь первый раз, сгенерируем user_id
        if user_id is None:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id

    recommendations_url = f"{recommendation_service_url}/recs/{user_id}"
    response = requests.get(recommendations_url)

    if response.status_code == 200:
        recommended_item_ids = response.json()['item_ids']
    else:
        # тут можно сделать fallback на стороне фронтенда
        recommended_item_ids = []

    print(recommended_item_ids)
    items_data = fetch_items_data_for_item_ids(recommended_item_ids)
    print(items_data)
    return render_template(
        'index.html',
        items_data=items_data,
        interactions_url=interactions_url
    )


def get_user_id_from_cookies():
    # Implement your logic to retrieve user_id from cookies
    return 'user_id'


def fetch_items_data_for_item_ids(item_ids):
    return [
               {
                   "item_id": item_id,
                   "imdb_url": movie_id_imdb[item_id],
                   "image_filename": f'{item_id}.jpg',
                   "title": movie_id_title[item_id]
               }
               for item_id in item_ids
               if item_id in movie_id_title
           ][:TOP_K]


if __name__ == '__main__':
    data = {
        "item_ids": list(map(str, movie_id_title.keys())),
    }
    requests.post(f'{recommendation_service_url}/add_items', json=data)
    app.run(host='0.0.0.0', port=8000, debug=True)
