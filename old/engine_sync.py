from lightfm import LightFM
from lightfm.datasets import fetch_movielens
import numpy as np, os
import polars as pl
from scipy.sparse import coo_matrix
import redis
from typing import List

redis_connection = redis.Redis('localhost')

current_dir = os.path.dirname(os.path.abspath(__file__))
interactions_path = os.path.join(current_dir, '..', 'data', 'interactions.csv')

def get_mapping(df, col):
    df = df.select(pl.col(col)).unique()
    df = df.with_row_count('int_map', offset=0)
    return {
        'forward': dict(zip(df[col], df['int_map'])),
        'backward': dict(zip(df['int_map'], df[col])),
    }


def save_mapping(mapping, bucket='users'):
    for direction in ['forward', 'backward']:
        redis_connection.json().set(f"mapping:{bucket}:{direction}", "$", mapping[direction])
    print(f'Mapping save to bucket {bucket}')
    return


def load_mapping(bucket='users'):
    result = {}

    for direction in ['forward', 'backward']:
        result[direction] = redis_connection.json().get(f"mapping:{bucket}:{direction}")

    return {
        'forward': result['forward'],
        'backward': result['backward'],
    }


def load_interactions():
    global interactions_path
    interact_summary = (pl.read_csv(interactions_path)
                        .with_columns(pl.when(pl.col('action') == 'like').then(1).otherwise(-1).alias('score'))
                        .groupby('user_id', 'item_id')
                        .agg(pl.sum('score')))

    user_mapping = get_mapping(interact_summary, 'user_id')
    item_mapping = get_mapping(interact_summary, 'item_id')

    save_mapping(user_mapping, bucket='users')
    save_mapping(item_mapping, bucket='items')

    interact_summary = (
        interact_summary.with_columns(pl.col("user_id").map_dict(user_mapping['forward']).alias('user_id'),
                                      pl.col("item_id").map_dict(item_mapping['forward']).alias('item_id'), ))
    users = interact_summary['user_id'].to_list()
    items = interact_summary['item_id'].to_list()
    scores = interact_summary['score'].to_list()

    N_USERS = max(users) + 1
    N_ITEMS = max(items) + 1

    interactions = coo_matrix((scores, (users, items)), shape=(N_USERS, N_ITEMS))
    print('Interaction loaded ...')
    return interactions


def train_model(data,
                learning_rate=0.05,
                no_components=30,
                random_state=42,
                epochs=30,
                num_threads=2,
                verbose=True):
    model = LightFM(loss='warp',
                    learning_rate=learning_rate,
                    no_components=no_components,
                    random_state=random_state)

    model.fit(data, epochs=epochs, num_threads=num_threads, verbose=verbose)
    print('Model built ...')
    return model


def get_top_n_items_for_all_users(model, interactions, n_items=10, num_threads=4):
    n_users, n_items_total = interactions.shape

    scores = model.predict(
        user_ids=np.repeat(np.arange(n_users), n_items_total),
        item_ids=np.tile(np.arange(n_items_total), n_users),
        num_threads=num_threads
    ).reshape(n_users, n_items_total)

    top_items = np.argsort(-scores, axis=1)[:, :n_items]

    recommendations = {}
    for user_id in range(n_users):
        recommendations[user_id] = top_items[user_id].tolist()

    print('Recommendations generated ...')
    return recommendations


def save_recommendations(recommendations):
    user_mapping = load_mapping(bucket='users')['backward']
    item_mapping = load_mapping(bucket='items')['backward']

    for user, rec in recommendations.items():
        user_origin = user_mapping[str(user)]
        rec_origin = [item_mapping[str(i)] for i in rec]
        key = f'recommendations:{user_origin}'
        redis_connection.delete(key)
        redis_connection.rpush(key, *rec_origin)
    print('Recommendations saved ...')
    return


def get_recommendation(user_id) -> List:
    key = f'recommendations:{user_id}'
    rec = redis_connection.get(key)
    return rec


def run(top_k=10):
    print('Start train process ...')
    # load coo matrix
    interactions = load_interactions()
    # LightFM model
    model = train_model(interactions)
    # build recommendations for each user
    recommendations = get_top_n_items_for_all_users(model, interactions, n_items=top_k)
    # save to redis
    save_recommendations(recommendations)
    return

if __name__ == '__main__':
    print('Run building ...')
    run()