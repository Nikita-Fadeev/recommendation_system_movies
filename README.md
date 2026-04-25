### Goal:
  Integration of Collaborative Filtering Model (ALS/LightFM) into a Production-Ready Backend. An educational project demonstrating the integration of a Collaborative Filtering model into a recommendation service backend.

### Features
  * Personalized recommendations for users based on their interaction history
  * Cold start handling: fallback to popular items
  * Core quality metrics: Precision@10, NDCG@10
  * Incremental model updates
  * Result caching to reduce latency

### Service structure:

├── ```event_collector```

│	   └── ```main.py``` # Stores user interaction history
  
├── ```recommendations```

│	   ├── ```data_proccess.py``` # Exports user-movie interaction data, saves training results to Redis
  
│    ├── ```engine.py``` # Trains recommendation models using collaborative filtering, computes recommendations for each known user

│    └── ```main.py``` # Returns recommendations to the frontend via user_id, optimizing NDCG, Diversity, and Coverage metrics

├── ```regular_pipeline```

│	    └── ```main.py``` # Task scheduler for cyclic model retraining
  
├── ```webapp```

│	   └── ```app.py``` Flask Frontend 
  
├── ```models.py``` # pydantic data models for validation

└── ```watched_filter.py``` # Watched content filtering module


### Metrics and result: 
  The recommendation service quality is assessed by an educational testing system.
  Evaluation criteria, baseline service metrics, and achieved results:

| Model     | Coverage | Diversity | NDCG@10 | Precision@10 |
|-----------|----------|-----------|---------|--------------|
| Baseline  | 0.65     | 0.28      | 0.0001  | 0.001        |
| Solution  | 1.00     | 0.80      | 0.049   | 0.500        |

### Tags
Python, Flask/FastAPI, Redis, Scikit-learn/LightFM/implicit, polars, scipy
