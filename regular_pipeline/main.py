import asyncio
import json
import os.path, os, sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aio_pika
import polars as pl
import redis
from aio_pika import Message
from recommendations.engine import ALSmodel as engine

redis_connection = redis.Redis('localhost')

current_dir = os.path.dirname(os.path.abspath(__file__))
interactions_path = os.path.join(current_dir, '..', 'data', 'interactions.csv')
print('interactions_path', interactions_path)

async def collect_messages():
    print('Start collecting messages from rabbitmq ...')
    connection = await aio_pika.connect_robust(
        "amqp://guest:guest@127.0.0.1/",
        loop=asyncio.get_event_loop()
    )

    queue_name = "user_interactions"
    routing_key = "user.interact.message"

    async with connection:
        print('Creating channel ...')
        channel = await connection.channel()

        print('Setting prefetch count to 10 ...')
        await channel.set_qos(prefetch_count=10)

        print(f'Declaring queue {queue_name} ...')
        queue = await channel.declare_queue(queue_name)

        print(f'Binding queue {queue_name} to exchange user.interact with routing key {routing_key} ...')
        exchange = await channel.declare_exchange("user.interact", type='direct')
        await queue.bind(exchange, routing_key)
        # await exchange.publish(Message(bytes(queue.name, "utf-8")), routing_key)

        t_start = time.time()
        data = []
        print('Start iterations ...')
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    message = message.body.decode()
                    if time.time() - t_start > 10:
                        print('saving events from rabbitmq')
                        # update data if 10s passed
                        df = pl.DataFrame(data)

                        if 'item_ids' in df.columns and 'actions' in df.columns:
                            new_data = df.explode(['item_ids', 'actions']).rename({
                                'item_ids': 'item_id',
                                'actions': 'action'
                            })
                        else:
                            print(f"Warning: Expected columns 'item_ids' and 'actions' not found. Available columns: {df.columns}")
                            new_data = df

                        if len(new_data) > 0:
                            if os.path.exists(interactions_path):
                                data = pl.concat([pl.read_csv(interactions_path), new_data])
                            else:
                                data = new_data
                            data.write_csv(interactions_path)

                        data = []
                        t_start = time.time()

                    message = json.loads(message)
                    data.append(message)


async def calculate_top_recommendations():
    while True:
        if os.path.exists(interactions_path):
            print('calculating top recommendations')
            interactions = pl.read_csv(interactions_path)
            top_items = (
                interactions
                .sort('timestamp')
                .unique(['user_id', 'item_id', 'action'], keep='last')
                .filter(pl.col('action') == 'like')
                .groupby('item_id')
                .count()
                .sort('count', descending=True)
                .head(100)
            )['item_id'].to_list()

            top_items = [str(item_id) for item_id in top_items]

            redis_connection.json().set('top_items', '.', top_items)
        await asyncio.sleep(10)


async def run_engine():
    last_mtime = 0

    while True:
        if os.path.exists(interactions_path):
            current_mtime = os.path.getmtime(interactions_path)

            if current_mtime > last_mtime:
                print(f"Data updated, restarting engine.run()...")
                last_mtime = current_mtime

                if asyncio.iscoroutinefunction(engine.run):
                    asyncio.create_task(engine.run())
                else:
                    asyncio.create_task(asyncio.to_thread(engine.run))

        await asyncio.sleep(10)

async def main():
    print('Start collecting messages and calculating recommendations ...')
    await asyncio.gather(
        collect_messages(),
        calculate_top_recommendations(),
        run_engine(),
    )

if __name__ == '__main__':
    print('Start main loop ...')
    asyncio.run(main())
