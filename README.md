# Single-Input Multiple-Output Model Merging: Leveraging Foundation Models for Multi-Task Learning 
TODO



## Benchmark
For standard 3 tasks in NYUv2 (without noise prediction task) in the multi-task learning setting with Split architecture, please follow the results below.


| Method                 | Sem. Seg. (mIOU) | Depth (aErr.) | Normal (mDist.) | Delta MTL | Norm |
|------------------------|------------------|---------------|-----------------|-----------|------|
| **MTL (Equal)**         |47.70            | 46.34         | -           | -      | - |
| **TS (Full Training)**  | 49.91            | 57.70         | -              | -         | - |
| **Sum-TA (Full Training)** | 20.04            | 105.20        | -              | -         | 1487 |
| **TS (Linear Probing)** | 52.06            | 46.88         | -              | -         | - |
| **Sum-TA (Linear Probing)** | 50.23            | 49.65         | -              | -         | 628 |

- **MTL**: Multi-Task Learning
- **TS**: Task-Specific
- **TA**: Task Arithmetic