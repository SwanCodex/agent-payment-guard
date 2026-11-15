# Step 7b: matched subset vs all rows with depth-balanced weights (LR on char n-grams)

Weight cap 20.0. Test split not used.

|                                                             |   C |   n_train |   ROC |    PR |   recall@0.1%FPR |   recall@0.5%FPR |   recall@1%FPR |
|:------------------------------------------------------------|----:|----------:|------:|------:|-----------------:|-----------------:|---------------:|
| ('A matched subset', 'val_matched')                         |  10 |     25768 | 0.918 | 0.908 |            0.021 |            0.139 |          0.302 |
| ('A matched subset', 'val_natural')                         |  10 |     25768 | 0.807 | 0.442 |            0.003 |            0.027 |          0.084 |
| ('A matched subset', 'external_phiusiil')                   |  10 |     25768 | 0.64  | 0.407 |            0.003 |            0.027 |          0.046 |
| ('B all rows, depth-balanced weights', 'val_matched')       |  10 |    122415 | 0.913 | 0.922 |            0.159 |            0.293 |          0.417 |
| ('B all rows, depth-balanced weights', 'val_natural')       |  10 |    122415 | 0.911 | 0.749 |            0.13  |            0.344 |          0.435 |
| ('B all rows, depth-balanced weights', 'external_phiusiil') |  10 |    122415 | 0.726 | 0.582 |            0.081 |            0.145 |          0.176 |
