import random
from typing import List

import numpy as np
import redis
from fastapi import FastAPI

from models import InteractEvent, RecommendationsResponse, NewItemsEvent
from watched_filter import WatchedFilter

app = FastAPI()

redis_connection = redis.Redis('localhost')
watched_filter = WatchedFilter()

unique_item_ids = set()
EPSILON = 0.05


@app.get('/healthcheck')
def healthcheck():
    return 200


@app.get('/cleanup')
def cleanup():
    global unique_item_ids
    unique_item_ids = set()
    try:
        redis_connection.delete('*')
        redis_connection.json().delete('*')
    except redis.exceptions.ConnectionError:
        pass
    return 200


@app.post('/add_items')
def add_movie(request: NewItemsEvent):
    global unique_item_ids
    for item_id in request.item_ids:
        unique_item_ids.add(item_id)
    return 200


@app.get('/recs/{user_id}')
def get_recs(user_id: str):
    global unique_item_ids

    try:
        most_popular = redis_connection.json().get('top_items')
    except redis.exceptions.ConnectionError:
        most_popular = []

    try:
        random_ids = np.random.choice(list(unique_item_ids), size=20, replace=False).tolist()
    except Exception:
        random_ids = []

    try:
        key = f'recomendations:{user_id}'
        personal_ids = [i.decode('utf-8') for i in redis_connection.lrange(key, 0, -1)]
    except redis.exceptions.ConnectionError:
        personal_ids = []

    candidates = list(set(personal_ids[:20] + most_popular[:5] + random_ids[:5]))
    sample_size = min(20, len(candidates))
    item_ids = np.random.choice(list(candidates), size=sample_size, replace=False)

    if random.random() < EPSILON:
        item_ids = np.random.choice(list(unique_item_ids), size=20, replace=False).tolist()

    return RecommendationsResponse(item_ids=item_ids)


@app.post('/interact')
async def interact(request: InteractEvent):
    watched_filter.add(request.user_id, request.item_id)
    return 200
