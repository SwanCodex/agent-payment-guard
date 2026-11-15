# Step 7c: weighted training on all train rows

Depth-balanced weights (cap 20.0). Choices made on val_natural PR-AUC. Test split not used.

|                                                          |   ROC |    PR |   recall@0.5%FPR |   recall@1%FPR |
|:---------------------------------------------------------|------:|------:|-----------------:|---------------:|
| ('LR n-grams', 'val_matched')                            | 0.913 | 0.922 |            0.307 |          0.443 |
| ('LR n-grams', 'val_natural')                            | 0.916 | 0.765 |            0.361 |          0.466 |
| ('LR n-grams', 'external_phiusiil')                      | 0.727 | 0.592 |            0.156 |          0.191 |
| ('LightGBM all feats', 'val_matched')                    | 0.794 | 0.799 |            0.184 |          0.212 |
| ('LightGBM all feats', 'val_natural')                    | 0.863 | 0.595 |            0.209 |          0.253 |
| ('LightGBM all feats', 'external_phiusiil')              | 0.338 | 0.216 |            0.004 |          0.007 |
| ('LightGBM no host feats', 'val_matched')                | 0.894 | 0.893 |            0.176 |          0.249 |
| ('LightGBM no host feats', 'val_natural')                | 0.881 | 0.704 |            0.294 |          0.406 |
| ('LightGBM no host feats', 'external_phiusiil')          | 0.424 | 0.285 |            0.028 |          0.047 |
| ('avg LR + LightGBM all feats', 'val_matched')           | 0.925 | 0.933 |            0.308 |          0.452 |
| ('avg LR + LightGBM all feats', 'val_natural')           | 0.932 | 0.805 |            0.411 |          0.529 |
| ('avg LR + LightGBM all feats', 'external_phiusiil')     | 0.689 | 0.54  |            0.122 |          0.15  |
| ('avg LR + LightGBM no host feats', 'val_matched')       | 0.942 | 0.945 |            0.413 |          0.499 |
| ('avg LR + LightGBM no host feats', 'val_natural')       | 0.942 | 0.835 |            0.491 |          0.588 |
| ('avg LR + LightGBM no host feats', 'external_phiusiil') | 0.65  | 0.501 |            0.122 |          0.145 |
