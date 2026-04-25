import asyncio, os
import redis.asyncio as redis_async
import redis
import polars as pl
from scipy.sparse import coo_matrix
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
interactions_path = os.path.join(current_dir, '..', 'data', 'interactions.csv')

redis_async_client = None
redis_sync_client = None

async def get_redis_async():
    global redis_async_client
    if redis_async_client is None:
        redis_async_client = await redis_async.from_url("redis://localhost")
    return redis_async_client


def get_redis_sync():
    global redis_sync_client
    if redis_sync_client is None:
        redis_sync_client = redis.Redis('localhost')
    return redis_sync_client


def get_mapping(df, col):
    df = df.select(pl.col(col)).unique()
    df = df.with_row_count('int_map', offset=0)
    return {
        'forward': dict(zip(df[col], df['int_map'])),
        'backward': dict(zip(df['int_map'], df[col])),
    }


def save_mapping(mapping, bucket='users'):
    r = get_redis_sync()
    for direction in ['forward', 'backward']:
        r.json().set(f"mapping:{bucket}:{direction}", "$", mapping[direction])
    print(f'Mapping save to bucket {bucket}')
    return


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


async def load_mapping(bucket='users'):
    r = await get_redis_async()
    result = {}

    for direction in ['forward', 'backward']:
        result[direction] = await r.json().get(f"mapping:{bucket}:{direction}")

    return {
        'forward': result['forward'],
        'backward': result['backward'],
    }


async def save_recomendations(recommendations):
    r = await get_redis_async()
    user_mapping_task = load_mapping(bucket='users')
    item_mapping_task = load_mapping(bucket='items')

    user_mapping = (await user_mapping_task)['backward']
    item_mapping = (await item_mapping_task)['backward']

    for user, rec in recommendations.items():
        user_origin = user_mapping[str(user)]
        rec_origin = [item_mapping[str(i)] for i in rec]
        key = f'recomendations:{user_origin}'
        await r.delete(key)
        await r.rpush(key, *rec_origin)
    print('Recomendations saved ...')
    return