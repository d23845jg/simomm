# Single-Input Multiple-Output Model Merging: Leveraging Foundation Models for Multi-Task Learning 
TODO



## Benchmark
For standard 3 tasks in NYUv2 (without noise prediction task) in the multi-task learning setting with Split architecture, please follow the results below.


| Method                 | Sem. Seg. (mIOU) | Depth (aErr.) | Normal (mDist.) | Delta MTL | Norm |
|------------------------|------------------|---------------|-----------------|-----------|------|
| **MTL (Equal)**         |47.70            | 46.34         | -           | -      | - |
| **TS FFT**  | 49.91            | 57.70         | -              | -         | - |
| **Sum-TA FFT ($\alpha=0.5$)** | 20.04            | 105.20        | -              | -         | 1487 |
| **TS LP** | 52.06            | 46.88         | -              | -         | - |
| **Sum-TA LP ($\alpha=0.5$)** | 35.53            | 59.71         | -              | -         | 440 |
| **Sum-TA LP + FFT ($\alpha=0.5$)** | 50.23            | 49.65         | -              | -         | 628 |

- **MTL**: Multi-Task Learning
- **TS**: Task-Specific
- **TA**: Task Arithmetic
- **FFT**: Full Fine-Tuning
- **LP**: Linear Probing