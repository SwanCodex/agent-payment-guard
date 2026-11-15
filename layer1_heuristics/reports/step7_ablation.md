# Step 7a: feature-group ablation

LightGBM (regularized: num_leaves 15, min_child_samples 200, lr 0.03, lambda_l2 10), trained on depth-matched train,
early stopping on val_matched. Test split not used.

|                             |   n_feat |   best_iter |   ROC matched |   ROC natural |   ROC external |   PR matched |   recall@0.5%FPR matched |
|:----------------------------|---------:|------------:|--------------:|--------------:|---------------:|-------------:|-------------------------:|
| all features                |       35 |          12 |         0.732 |         0.505 |          0.364 |        0.73  |                    0.082 |
| drop host                   |       26 |          40 |         0.795 |         0.534 |          0.352 |        0.78  |                    0.114 |
| drop path                   |       27 |          10 |         0.733 |         0.573 |          0.503 |        0.716 |                    0.027 |
| drop urlwide                |       28 |          10 |         0.655 |         0.436 |          0.412 |        0.675 |                    0.055 |
| drop flags                  |       30 |          17 |         0.757 |         0.543 |          0.369 |        0.754 |                    0.075 |
| drop lexicon                |       30 |          17 |         0.742 |         0.531 |          0.373 |        0.743 |                    0.08  |
| drop tld                    |       34 |          17 |         0.732 |         0.492 |          0.363 |        0.73  |                    0.077 |
| drop path_depth+is_homepage |       33 |          11 |         0.74  |         0.532 |          0.499 |        0.736 |                    0.08  |
| only host                   |        9 |           8 |         0.616 |         0.648 |          0.482 |        0.647 |                    0.008 |
| only path                   |        8 |          53 |         0.76  |         0.522 |          0.445 |        0.742 |                    0.107 |
| only urlwide                |        7 |           1 |         0.741 |         0.584 |          0.404 |        0.706 |                    0.019 |
| only flags                  |        5 |           1 |         0.542 |         0.542 |          0.501 |        0.542 |                    0.088 |
| only lexicon                |        5 |           1 |         0.484 |         0.475 |          0.478 |        0.492 |                    0.005 |
| only tld                    |        1 |           4 |         0.68  |         0.659 |          0.584 |        0.647 |                    0.033 |
