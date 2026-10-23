| Model | Training data | Accuracy | Macro-F1 | Top-3 acc. | Acc. with 5% typos | Typo drop | Latency p50 / p95 (ms) | Device |
|---|---|---|---|---|---|---|---|---|
| TF-IDF (word+char) + LogReg | 10% (900) | 75.6% | 75.2% | 90.7% | 69.0% | -6.7 pts | 2.3 / 3.4 | cpu |
| TF-IDF (word+char) + LogReg | 25% (2,251) | 84.6% | 84.6% | 94.6% | 78.5% | -6.2 pts | 2.2 / 3.6 | cpu |
| TF-IDF (word+char) + LogReg | 50% (4,502) | 88.4% | 88.4% | 96.4% | 83.1% | -5.3 pts | 2.7 / 3.8 | cpu |
| TF-IDF (word+char) + LogReg | 100% (9,003) | 91.0% | 91.0% | 97.3% | 85.6% | -5.4 pts | 3.1 / 4.1 | cpu |
