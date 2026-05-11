## Results

<!-- RESULTS:START -->

_N examples: 51; contract failures: 0_

| metric                      | raw                  | wrapped              |
| --------------------------- | -------------------- | -------------------- |
| hallucination rate (95% CI) | 0.078 [0.020, 0.157] | 0.160 [0.080, 0.260] |
| ECE (10 bins)               | 0.078                | 0.073                |
| Brier                       | 0.078                | 0.098                |
| accuracy on judged spans    | 0.922                | 0.840                |
| spans (judged / abstain)    | 51 / 0               | 50 / 13              |

Selective accuracy (wrapped) at coverage:

| coverage | accuracy |
| -------- | -------- |
| 25%      | 0.923    |
| 50%      | 0.920    |
| 75%      | 0.947    |
| 100%     | 0.840    |

Reliability diagram (wrapped): ![reliability](reliability.png)

ASCII reliability (wrapped):

```
bin     n   conf   acc
0.2-0.3   1  0.30  0.00 |
0.5-0.6   2  0.50  0.00 |
0.8-0.9   2  0.82  0.50 |##########
0.9-1.0  45  0.95  0.91 |##################
```

<!-- RESULTS:END -->
