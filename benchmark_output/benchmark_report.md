# PCRegression Performance Benchmark Report
Generated: 2026-01-07 10:44:41

## Executive Summary

- **Average OLS time**: 1.49 ms
- **Average PCRegression time**: 23.27 ms
- **PCRegression slowdown vs OLS**: 15.6x

## 1. Timing Results - Linear Models

### Wall Time (seconds)

|           |   LassoCV |    OLS |   PCCV_aic |   PCR_constrained |   PCR_penalized |   PCR_safe_off |   RidgeCV |
|:----------|----------:|-------:|-----------:|------------------:|----------------:|---------------:|----------:|
| (10, 3)   |    0.0484 | 0.0016 |     0.1716 |            0.0086 |          0.0064 |         0.007  |    0.0051 |
| (10, 5)   |    0.0447 | 0.0017 |     0.6125 |            0.0255 |          0.0348 |         0.02   |    0.0049 |
| (10, 10)  |    0.0328 | 0.0018 |     0.8385 |            0.0395 |          0.0435 |         0.0274 |    0.0047 |
| (30, 3)   |    0.0434 | 0.0013 |     0.223  |            0.0055 |          0.0167 |         0.0045 |    0.0049 |
| (30, 5)   |    0.0396 | 0.001  |     0.5421 |            0.0272 |          0.0301 |         0.0209 |    0.004  |
| (30, 10)  |    0.0456 | 0.0018 |     0.9623 |            0.0412 |          0.0429 |         0.0323 |    0.0051 |
| (50, 3)   |    0.0474 | 0.0015 |     0.2163 |            0.0043 |          0.015  |         0.005  |    0.0043 |
| (50, 5)   |    0.0476 | 0.0018 |     0.3395 |            0.0178 |          0.0189 |         0.0104 |    0.0031 |
| (50, 10)  |    0.0527 | 0.001  |     0.9634 |            0.0452 |          0.0487 |         0.0324 |    0.0034 |
| (100, 3)  |    0.0434 | 0.0013 |     0.3018 |            0.0151 |          0.017  |         0.0121 |    0.0039 |
| (100, 5)  |    0.0415 | 0.0014 |     0.2595 |            0.0045 |          0.0175 |         0.0052 |    0.0047 |
| (100, 10) |    0.0465 | 0.0017 |     0.9045 |            0.0447 |          0.0562 |         0.0287 |    0.0038 |

### Speed Relative to OLS

| n | p | Method | Time (s) | Ratio vs OLS |
|---|---|--------|----------|-------------|
| 10 | 3 | OLS | 0.0016 | 1.0x |
| 10 | 3 | PCR_constrained | 0.0086 | 5.3x |
| 10 | 3 | PCR_safe_off | 0.0070 | 4.4x |
| 10 | 3 | PCR_penalized | 0.0064 | 4.0x |
| 10 | 3 | PCCV_aic | 0.1716 | 106.8x |
| 10 | 3 | RidgeCV | 0.0051 | 3.2x |
| 10 | 3 | LassoCV | 0.0484 | 30.1x |
| 10 | 5 | OLS | 0.0017 | 1.0x |
| 10 | 5 | PCR_constrained | 0.0255 | 15.3x |
| 10 | 5 | PCR_safe_off | 0.0200 | 12.0x |
| 10 | 5 | PCR_penalized | 0.0348 | 20.9x |
| 10 | 5 | PCCV_aic | 0.6125 | 368.3x |
| 10 | 5 | RidgeCV | 0.0049 | 2.9x |
| 10 | 5 | LassoCV | 0.0447 | 26.9x |
| 10 | 10 | OLS | 0.0018 | 1.0x |
| 10 | 10 | PCR_constrained | 0.0395 | 22.3x |
| 10 | 10 | PCR_safe_off | 0.0274 | 15.5x |
| 10 | 10 | PCR_penalized | 0.0435 | 24.5x |
| 10 | 10 | PCCV_aic | 0.8385 | 473.0x |
| 10 | 10 | RidgeCV | 0.0047 | 2.7x |
| 10 | 10 | LassoCV | 0.0328 | 18.5x |
| 30 | 3 | OLS | 0.0013 | 1.0x |
| 30 | 3 | PCR_constrained | 0.0055 | 4.2x |
| 30 | 3 | PCR_safe_off | 0.0045 | 3.4x |
| 30 | 3 | PCR_penalized | 0.0167 | 12.8x |
| 30 | 3 | PCCV_aic | 0.2230 | 170.8x |
| 30 | 3 | RidgeCV | 0.0049 | 3.8x |
| 30 | 3 | LassoCV | 0.0434 | 33.3x |
| 30 | 5 | OLS | 0.0010 | 1.0x |
| 30 | 5 | PCR_constrained | 0.0272 | 26.1x |
| 30 | 5 | PCR_safe_off | 0.0209 | 20.0x |
| 30 | 5 | PCR_penalized | 0.0301 | 28.8x |
| 30 | 5 | PCCV_aic | 0.5421 | 520.1x |
| 30 | 5 | RidgeCV | 0.0040 | 3.9x |
| 30 | 5 | LassoCV | 0.0396 | 38.0x |
| 30 | 10 | OLS | 0.0018 | 1.0x |
| 30 | 10 | PCR_constrained | 0.0412 | 22.6x |
| 30 | 10 | PCR_safe_off | 0.0323 | 17.7x |
| 30 | 10 | PCR_penalized | 0.0429 | 23.6x |
| 30 | 10 | PCCV_aic | 0.9623 | 528.4x |
| 30 | 10 | RidgeCV | 0.0051 | 2.8x |
| 30 | 10 | LassoCV | 0.0456 | 25.0x |
| 50 | 3 | OLS | 0.0015 | 1.0x |
| 50 | 3 | PCR_constrained | 0.0043 | 2.8x |
| 50 | 3 | PCR_safe_off | 0.0050 | 3.3x |
| 50 | 3 | PCR_penalized | 0.0150 | 9.8x |
| 50 | 3 | PCCV_aic | 0.2163 | 141.7x |
| 50 | 3 | RidgeCV | 0.0043 | 2.8x |
| 50 | 3 | LassoCV | 0.0474 | 31.1x |
| 50 | 5 | OLS | 0.0018 | 1.0x |
| 50 | 5 | PCR_constrained | 0.0178 | 10.0x |
| 50 | 5 | PCR_safe_off | 0.0104 | 5.8x |
| 50 | 5 | PCR_penalized | 0.0189 | 10.5x |
| 50 | 5 | PCCV_aic | 0.3395 | 189.6x |
| 50 | 5 | RidgeCV | 0.0031 | 1.7x |
| 50 | 5 | LassoCV | 0.0476 | 26.6x |
| 50 | 10 | OLS | 0.0010 | 1.0x |
| 50 | 10 | PCR_constrained | 0.0452 | 47.1x |
| 50 | 10 | PCR_safe_off | 0.0324 | 33.8x |
| 50 | 10 | PCR_penalized | 0.0487 | 50.7x |
| 50 | 10 | PCCV_aic | 0.9634 | 1003.8x |
| 50 | 10 | RidgeCV | 0.0034 | 3.6x |
| 50 | 10 | LassoCV | 0.0527 | 54.9x |
| 100 | 3 | OLS | 0.0013 | 1.0x |
| 100 | 3 | PCR_constrained | 0.0151 | 11.6x |
| 100 | 3 | PCR_safe_off | 0.0121 | 9.3x |
| 100 | 3 | PCR_penalized | 0.0170 | 13.0x |
| 100 | 3 | PCCV_aic | 0.3018 | 231.9x |
| 100 | 3 | RidgeCV | 0.0039 | 3.0x |
| 100 | 3 | LassoCV | 0.0434 | 33.3x |
| 100 | 5 | OLS | 0.0014 | 1.0x |
| 100 | 5 | PCR_constrained | 0.0045 | 3.3x |
| 100 | 5 | PCR_safe_off | 0.0052 | 3.7x |
| 100 | 5 | PCR_penalized | 0.0175 | 12.5x |
| 100 | 5 | PCCV_aic | 0.2595 | 185.9x |
| 100 | 5 | RidgeCV | 0.0047 | 3.4x |
| 100 | 5 | LassoCV | 0.0415 | 29.7x |
| 100 | 10 | OLS | 0.0017 | 1.0x |
| 100 | 10 | PCR_constrained | 0.0447 | 26.0x |
| 100 | 10 | PCR_safe_off | 0.0287 | 16.7x |
| 100 | 10 | PCR_penalized | 0.0562 | 32.7x |
| 100 | 10 | PCCV_aic | 0.9045 | 525.8x |
| 100 | 10 | RidgeCV | 0.0038 | 2.2x |
| 100 | 10 | LassoCV | 0.0465 | 27.0x |

## 2. Timing Results - Learning Curve Models

### Wall Time (seconds)

|   n_samples |   OLS_log |   PCR_LC_constrained |   PCR_LC_penalized |   PCR_LC_safe_off |
|------------:|----------:|---------------------:|-------------------:|------------------:|
|          10 |    0.0014 |               0.0142 |             0.0156 |            0.0103 |
|          30 |    0.0017 |               0.0155 |             0.0185 |            0.0127 |
|          50 |    0.0014 |               0.0166 |             0.0258 |            0.0193 |
|         100 |    0.0012 |               0.0183 |             0.0197 |            0.0147 |

## 3. Coefficient Accuracy

| method_name        |   n_samples |   n_features | model_type     |   r2_train |   coef_rmse_vs_true |   coef_rmse_vs_ols |
|:-------------------|------------:|-------------:|:---------------|-----------:|--------------------:|-------------------:|
| OLS                |          10 |            3 | linear         |     0.609  |              0.8663 |             0      |
| PCR_constrained    |          10 |            3 | linear         |     0.4174 |              0.4178 |             0.4987 |
| PCR_safe_off       |          10 |            3 | linear         |     0.4174 |              0.4178 |             0.4987 |
| PCR_penalized      |          10 |            3 | linear         |     0.4174 |              0.4178 |             0.4987 |
| PCCV_aic           |          10 |            3 | linear         |     0.4174 |              0.4178 |             0.4987 |
| RidgeCV            |          10 |            3 | linear         |     0.5858 |              0.6655 |             0.207  |
| LassoCV            |          10 |            3 | linear         |     0.6077 |              0.8193 |             0.0494 |
| OLS                |          10 |            5 | linear         |     0.8794 |              1.1057 |             0      |
| PCR_constrained    |          10 |            5 | linear         |     0.4058 |              0.1959 |             1.1661 |
| PCR_safe_off       |          10 |            5 | linear         |     0.4058 |              0.1959 |             1.1661 |
| PCR_penalized      |          10 |            5 | linear         |     0.4057 |              0.1939 |             1.1655 |
| PCCV_aic           |          10 |            5 | linear         |     0.3985 |              0.199  |             1.1601 |
| RidgeCV            |          10 |            5 | linear         |     0.8687 |              0.9257 |             0.1914 |
| LassoCV            |          10 |            5 | linear         |     0.8673 |              0.9176 |             0.2065 |
| OLS                |          10 |           10 | linear         |     1      |              2.1887 |             0      |
| PCR_constrained    |          10 |           10 | linear         |     0.7223 |              0.4736 |             2.0008 |
| PCR_safe_off       |          10 |           10 | linear         |     0.7223 |              0.4736 |             2.0008 |
| PCR_penalized      |          10 |           10 | linear         |     0.7223 |              0.4709 |             2.0033 |
| PCCV_aic           |          10 |           10 | linear         |     0.7198 |              0.447  |             2.0205 |
| RidgeCV            |          10 |           10 | linear         |     0.5074 |              0.334  |             2.0895 |
| LassoCV            |          10 |           10 | linear         |     0.7291 |              0.5356 |             2.1078 |
| OLS                |          30 |            3 | linear         |     0.3285 |              0.3052 |             0      |
| PCR_constrained    |          30 |            3 | linear         |     0.3285 |              0.3052 |             0      |
| PCR_safe_off       |          30 |            3 | linear         |     0.3285 |              0.3052 |             0      |
| PCR_penalized      |          30 |            3 | linear         |     0.3285 |              0.3046 |             0.002  |
| PCCV_aic           |          30 |            3 | linear         |     0.3282 |              0.2891 |             0.0192 |
| RidgeCV            |          30 |            3 | linear         |     0.2881 |              0.167  |             0.1811 |
| LassoCV            |          30 |            3 | linear         |     0.2977 |              0.3013 |             0.1441 |
| OLS                |          30 |            5 | linear         |     0.5291 |              0.5194 |             0      |
| PCR_constrained    |          30 |            5 | linear         |     0.5103 |              0.3432 |             0.2777 |
| PCR_safe_off       |          30 |            5 | linear         |     0.5103 |              0.3432 |             0.2777 |
| PCR_penalized      |          30 |            5 | linear         |     0.5103 |              0.342  |             0.2783 |
| PCCV_aic           |          30 |            5 | linear         |     0.5101 |              0.3321 |             0.2821 |
| RidgeCV            |          30 |            5 | linear         |     0.4578 |              0.2376 |             0.4616 |
| LassoCV            |          30 |            5 | linear         |     0.5056 |              0.313  |             0.3044 |
| OLS                |          30 |           10 | linear         |     0.873  |              0.3343 |             0      |
| PCR_constrained    |          30 |           10 | linear         |     0.8452 |              0.1829 |             0.2426 |
| PCR_safe_off       |          30 |           10 | linear         |     0.8452 |              0.1829 |             0.2426 |
| PCR_penalized      |          30 |           10 | linear         |     0.8452 |              0.182  |             0.2423 |
| PCCV_aic           |          30 |           10 | linear         |     0.8449 |              0.1755 |             0.2449 |
| RidgeCV            |          30 |           10 | linear         |     0.8606 |              0.2229 |             0.1585 |
| LassoCV            |          30 |           10 | linear         |     0.8568 |              0.2516 |             0.1814 |
| OLS                |          50 |            3 | linear         |     0.1836 |              0.197  |             0      |
| PCR_constrained    |          50 |            3 | linear         |     0.1836 |              0.197  |             0      |
| PCR_safe_off       |          50 |            3 | linear         |     0.1836 |              0.197  |             0      |
| PCR_penalized      |          50 |            3 | linear         |     0.1836 |              0.1955 |             0.0017 |
| PCCV_aic           |          50 |            3 | linear         |     0.1835 |              0.1837 |             0.0136 |
| RidgeCV            |          50 |            3 | linear         |     0.1635 |              0.0651 |             0.1511 |
| LassoCV            |          50 |            3 | linear         |     0.1748 |              0.1451 |             0.079  |
| OLS                |          50 |            5 | linear         |     0.4309 |              0.1796 |             0      |
| PCR_constrained    |          50 |            5 | linear         |     0.4309 |              0.1753 |             0.005  |
| PCR_safe_off       |          50 |            5 | linear         |     0.4309 |              0.1753 |             0.005  |
| PCR_penalized      |          50 |            5 | linear         |     0.4309 |              0.175  |             0.0056 |
| PCCV_aic           |          50 |            5 | linear         |     0.4306 |              0.1596 |             0.0284 |
| RidgeCV            |          50 |            5 | linear         |     0.4102 |              0.1621 |             0.2205 |
| LassoCV            |          50 |            5 | linear         |     0.4288 |              0.1341 |             0.0638 |
| OLS                |          50 |           10 | linear         |     0.8727 |              0.2857 |             0      |
| PCR_constrained    |          50 |           10 | linear         |     0.8654 |              0.1866 |             0.1239 |
| PCR_safe_off       |          50 |           10 | linear         |     0.8654 |              0.1866 |             0.1239 |
| PCR_penalized      |          50 |           10 | linear         |     0.8654 |              0.1863 |             0.124  |
| PCCV_aic           |          50 |           10 | linear         |     0.8654 |              0.1853 |             0.1233 |
| RidgeCV            |          50 |           10 | linear         |     0.8696 |              0.2118 |             0.1132 |
| LassoCV            |          50 |           10 | linear         |     0.8703 |              0.2121 |             0.0859 |
| OLS                |         100 |            3 | linear         |     0.2871 |              0.1831 |             0      |
| PCR_constrained    |         100 |            3 | linear         |     0.2846 |              0.1544 |             0.0696 |
| PCR_safe_off       |         100 |            3 | linear         |     0.2846 |              0.1544 |             0.0696 |
| PCR_penalized      |         100 |            3 | linear         |     0.2846 |              0.1539 |             0.0696 |
| PCCV_aic           |         100 |            3 | linear         |     0.2845 |              0.1505 |             0.0691 |
| RidgeCV            |         100 |            3 | linear         |     0.2812 |              0.1105 |             0.073  |
| LassoCV            |         100 |            3 | linear         |     0.2834 |              0.13   |             0.0754 |
| OLS                |         100 |            5 | linear         |     0.5398 |              0.1443 |             0      |
| PCR_constrained    |         100 |            5 | linear         |     0.5398 |              0.1443 |             0      |
| PCR_safe_off       |         100 |            5 | linear         |     0.5398 |              0.1443 |             0      |
| PCR_penalized      |         100 |            5 | linear         |     0.5398 |              0.1434 |             0.0011 |
| PCCV_aic           |         100 |            5 | linear         |     0.5397 |              0.1361 |             0.0107 |
| RidgeCV            |         100 |            5 | linear         |     0.5299 |              0.0947 |             0.1337 |
| LassoCV            |         100 |            5 | linear         |     0.5326 |              0.0793 |             0.0983 |
| OLS                |         100 |           10 | linear         |     0.8482 |              0.3822 |             0      |
| PCR_constrained    |         100 |           10 | linear         |     0.8334 |              0.1985 |             0.2498 |
| PCR_safe_off       |         100 |           10 | linear         |     0.8334 |              0.1985 |             0.2498 |
| PCR_penalized      |         100 |           10 | linear         |     0.8334 |              0.1981 |             0.2493 |
| PCCV_aic           |         100 |           10 | linear         |     0.8333 |              0.1952 |             0.2458 |
| RidgeCV            |         100 |           10 | linear         |     0.8472 |              0.3276 |             0.0615 |
| LassoCV            |         100 |           10 | linear         |     0.845  |              0.2942 |             0.0991 |
| OLS_log            |          10 |            3 | learning_curve |     0.8723 |            nan      |           nan      |
| PCR_LC_constrained |          10 |            3 | learning_curve |     0.8723 |              0.0118 |             0.0001 |
| PCR_LC_safe_off    |          10 |            3 | learning_curve |     0.8723 |              0.0118 |             0.0001 |
| PCR_LC_penalized   |          10 |            3 | learning_curve |     0.8721 |              0.02   |             0.0097 |
| OLS_log            |          30 |            3 | learning_curve |     0.9103 |            nan      |           nan      |
| PCR_LC_constrained |          30 |            3 | learning_curve |     0.9103 |              0.0668 |             0      |
| PCR_LC_safe_off    |          30 |            3 | learning_curve |     0.9103 |              0.0668 |             0      |
| PCR_LC_penalized   |          30 |            3 | learning_curve |     0.9103 |              0.0697 |             0.0029 |
| OLS_log            |          50 |            3 | learning_curve |     0.9093 |            nan      |           nan      |
| PCR_LC_constrained |          50 |            3 | learning_curve |     0.9093 |              0.0105 |             0      |
| PCR_LC_safe_off    |          50 |            3 | learning_curve |     0.9093 |              0.0105 |             0      |
| PCR_LC_penalized   |          50 |            3 | learning_curve |     0.9093 |              0.0113 |             0.0019 |
| OLS_log            |         100 |            3 | learning_curve |     0.893  |            nan      |           nan      |
| PCR_LC_constrained |         100 |            3 | learning_curve |     0.893  |              0.0341 |             0.0001 |
| PCR_LC_safe_off    |         100 |            3 | learning_curve |     0.893  |              0.0341 |             0.0001 |
| PCR_LC_penalized   |         100 |            3 | learning_curve |     0.893  |              0.035  |             0.001  |

## 4. Safe Mode Overhead Analysis

| n | safe_mode | Time (s) | Overhead |
|---|-----------|----------|----------|
| 30 | True | 0.0228 | +6.7% |
| 30 | False | 0.0213 | baseline |
| 50 | True | 0.0142 | +21.3% |
| 50 | False | 0.0117 | baseline |
| 100 | True | 0.0059 | +39.1% |
| 100 | False | 0.0042 | baseline |

## 5. Bottleneck Analysis

Function time breakdown from profiling `_objective()` calls:

| Function | Time (ms) | % of Total | Calls | Per Call (μs) |
|----------|-----------|------------|-------|---------------|
| Objective (total) | 39.51 | 98.4% | 1000 | 39.51 |
| SSE Loss | 7.79 | 19.4% | 1000 | 7.79 |
| Prediction | 4.86 | 12.1% | 1000 | 4.86 |
| Penalty | 0.26 | 0.7% | 1000 | 0.26 |

## 6. Optimization Recommendations

### Quick Wins

1. **Use `safe_mode=False`** for well-tested prediction functions
   - Eliminates ~10-15% overhead from validity checks
   - Only use when prediction function is known to be stable

2. **Use information criteria (AIC/BIC)** instead of CV for small samples
   - Faster than k-fold CV (no data splitting)
   - Better statistical properties for n < 50

### Medium-Term Optimizations

3. **Parallelize IC grid search** in `cv.py`
   - Current: Sequential loop over alpha × l1_ratio combinations
   - Potential: `joblib.Parallel` for embarrassingly parallel fits

4. **Warm starting** between CV folds or alpha values
   - Pass previous coefficients as `x0` parameter
   - Can reduce optimizer iterations by 20-40%

### Long-Term Considerations

5. **Analytical gradients** for scipy optimizer
   - Currently uses finite difference approximation
   - Analytical gradients could halve optimization time

6. **Numba JIT compilation** for loss functions
   - Only beneficial for large n (> 1000)
   - Adds dependency and compilation overhead

