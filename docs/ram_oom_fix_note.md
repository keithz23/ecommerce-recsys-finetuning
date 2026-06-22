# RAM OOM Fix Note

## Context
Notebook `notebooks/combined_recommendation_system.ipynb` was crashing on Colab and Kaggle because the dataset is too large for operations that expand sparse interactions into dense structures.

Current data scale observed during inspection:
- `data/input/interaction.csv`: about 18.7 million rows
- Unique users: about 9.7 million
- Unique products: about 1.47 million
- A dense user-item matrix at this scale is infeasible

## What was changed

### 1. Removed dense pivot-table creation in recommendation generation
File: [src/model_eval_functions.py](../src/model_eval_functions.py)

The `get_recommendations()` function originally built a full `pivot_table` of `user_id x prod_id` before finding items a user had not interacted with.

This was changed to:
- collect the user's interacted products directly from the input DataFrame
- build the candidate product list from unique product IDs
- avoid materializing a dense user-item matrix

### 2. Reduced parallel worker pressure in Surprise cross-validation and grid search
File: [src/cf_recommender.py](../src/cf_recommender.py)

The notebook used `n_jobs=-1` for:
- `cross_validate(...)`
- `GridSearchCV(...)`

This was changed to `n_jobs=1` so the process does not multiply memory usage across workers.

## Why these changes were needed

The original implementation was memory-heavy in two ways:
- dense pivot creation can explode RAM usage when user and product cardinality are large
- parallel CV/grid search can replicate data and model state across multiple processes

On Colab and Kaggle, that combination is enough to trigger out-of-memory crashes.

## What would happen if we did not change it

If the notebook stayed as it was:
- recommendation calls would keep building a huge dense matrix and likely crash mid-execution
- cross-validation and tuning would consume more RAM than available on typical Colab/Kaggle runtimes
- the notebook would remain unstable even if the data loading step itself succeeded

## Validation performed

- Syntax check passed for [src/model_eval_functions.py](../src/model_eval_functions.py)
- Syntax check passed for [src/cf_recommender.py](../src/cf_recommender.py)
- Search confirmed the dense pivot and `n_jobs=-1` hot spots were removed from the touched code paths

## Practical outcome

The notebook should now be significantly less likely to crash due to RAM pressure during recommendation generation and Surprise model evaluation.
