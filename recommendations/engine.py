import asyncio, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csr_matrix

from lightfm import LightFM
from implicit.als import AlternatingLeastSquares
from .data_proccess import load_interactions, save_recomendations


class LightFMmodel():

    @staticmethod
    def train_model(data,
                    learning_rate=0.01,
                    no_components=32,
                    random_state=42,
                    epochs=200,
                    num_threads=2,
                    verbose=True):

        model = LightFM(loss='warp',
                        learning_rate=learning_rate,
                        no_components=no_components,
                        random_state=random_state)

        model.fit(data, epochs=epochs, num_threads=num_threads, verbose=verbose)
        print('LightFM Model builded ...')
        return model

    @staticmethod
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

        print('Recomendations generated ...')
        return recommendations

    @staticmethod
    async def run(top_k=20):
        print('Start train process ...')
        executor = ThreadPoolExecutor(max_workers=2)
        loop = asyncio.get_event_loop()
        interactions = await loop.run_in_executor(executor, load_interactions)
        model = await loop.run_in_executor(executor, lambda: LightFMmodel.train_model(interactions))
        recommendations = await loop.run_in_executor(executor, lambda: LightFMmodel.get_top_n_items_for_all_users(model, interactions,
                                                                                                     n_items=top_k))
        await save_recomendations(recommendations)
        return


class ALSmodel():

    @staticmethod
    def get_top_n_items_for_all_users(model, user_items_matrix, n_items=10,
                                      filter_already_liked=True):

        if not isinstance(user_items_matrix, csr_matrix):
            user_items_matrix = user_items_matrix.tocsr()

        n_users, n_items_total = user_items_matrix.shape

        recommendations = {}

        for user_id in range(n_users):
            recommended_items = model.recommend(
                user_id,
                user_items_matrix[user_id],
                N=n_items,
                filter_already_liked_items=filter_already_liked,
                recalculate_user=False,
            )

            recommendations[user_id] = recommended_items[0].tolist()

            if (user_id + 1) % 1000 == 0:
                print(f'Processed {user_id + 1}/{n_users} users...')

        print(f'Recommendations generated for {n_users} users')
        return recommendations

    @staticmethod
    def train_model(data,
                    factors=25,
                    regularization=0.01,
                    iterations=40,
                    random_state=42,
                    use_gpu=False):

        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        random_state=random_state,
                                        use_gpu=use_gpu)

        model.fit(data)
        print('ALS Model builded ...')
        return model

    @staticmethod
    async def run(top_k=10):
        print('Start train process ...')
        executor = ThreadPoolExecutor(max_workers=2)
        loop = asyncio.get_event_loop()
        interactions = await loop.run_in_executor(executor, load_interactions)
        model = await loop.run_in_executor(executor, lambda: ALSmodel.train_model(interactions))
        recommendations = await loop.run_in_executor(executor,
                                                     lambda: ALSmodel.get_top_n_items_for_all_users(model, interactions,
                                                                                                    n_items=top_k))
        await save_recomendations(recommendations)
        return
