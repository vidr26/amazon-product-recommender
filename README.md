# Amazon Product Recommender

Recommends electronics products to Amazon customers from their past ratings. It
compares rank-based, user-user, item-item and matrix-factorization approaches.

Completed as part of the **Applied Data Science Program**, MIT Professional Education.

- **Notebook (code, analysis and conclusions):** [`VidyaVasist_Elective_project_RecommendationSystems.ipynb`](VidyaVasist_Elective_project_RecommendationSystems.ipynb)

## Problem

Online retailers rely on personalized recommendations to help customers find
products in very large catalogs. The goal was to recommend products to each
user from ratings alone, with no product descriptions or review text.

## Approach

1. **Data reduction.** The full dataset has 7.8M ratings. It was filtered to
   users with at least 50 ratings and products with at least 5, leaving 65,290
   ratings from 1,540 users on 5,689 products.
2. **Rank-based model.** Top products by average rating, with a minimum number
   of interactions, for new users who have no history (the cold-start problem).
3. **Collaborative filtering.** User-user and item-item KNN similarity models.
4. **Matrix factorization.** SVD, which learns latent user and product factors.
5. **Tuning and evaluation.** `GridSearchCV` for each model. Scored with RMSE
   and with precision, recall and F1 at k = 10, counting a product as relevant
   at a rating of 3.5 or above.

## Results

| Model | RMSE | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|---|
| User-user (baseline) | 1.001 | 0.855 | 0.858 | 0.856 |
| **User-user (tuned)** | 0.953 | 0.847 | **0.894** | **0.870** |
| Item-item (baseline) | 0.995 | 0.838 | 0.845 | 0.841 |
| Item-item (tuned) | 0.958 | 0.839 | 0.880 | 0.859 |
| SVD (baseline) | 0.888 | 0.853 | 0.880 | 0.866 |
| **SVD (tuned)** | **0.881** | **0.854** | 0.878 | 0.866 |

Tuned SVD predicts ratings most accurately (lowest RMSE). Tuned user-user
filtering produces the most relevant top-10 lists (highest F1). Tuning improved
both similarity models noticeably, but barely changed SVD.

## Data

[Amazon product ratings, Electronics category](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
(`ratings_Electronics.csv`: user ID, product ID, rating, timestamp). The
notebook was run in Google Colab. To rerun it elsewhere, change the
`pd.read_csv` path to the file's location.

## Tools

Python, Pandas, NumPy, Surprise, Matplotlib, Seaborn
