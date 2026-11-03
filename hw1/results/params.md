| Model | best params (by val NDCG@10) | val NDCG@10 |
|---|---|---|
| ALS (best: implicit) | `f=64, iterations=15, lam=10.0, implicit=1, alpha=0.0` | 0.020777279419652062 |
| FM (meta) | `objective=bpr, dim=32, lr=0.02, epochs=6, batch_size=8192, eval_every=2, feature_set=full, wd=0.0001` | 0.0144746301619495 |
| Popularity | `` | 0.01668220558727284 |
| EASE | `lam=4000` | 0.018247396031090025 |
| timeSVD++ | `f=32, lr=0.005, epochs=14, bin_days=90, batch_size=8192, eval_every=2` | 0.013678498617286712 |
| FM (IDs only) | `objective=bpr, dim=16, lr=0.01, epochs=8, batch_size=8192, eval_every=2, feature_set=ids, wd=0.0001` | 0.01124860945653953 |
| Item-KNN | `shrinkage=100, topk=100` | 0.015885975106904253 |
| Random | `` | 0.00035417643577078403 |
