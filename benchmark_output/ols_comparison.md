# OLS vs PCRegression Comparison

## Speed Comparison

### n=10, p=3

- **OLS**: 1.61ms (1.0x OLS), R²=0.6090, RMSE vs true=0.8663
- **PCR_constrained**: 8.58ms (5.3x OLS), R²=0.4174, RMSE vs true=0.4178
- **PCR_penalized**: 6.39ms (4.0x OLS), R²=0.4174, RMSE vs true=0.4178
- **PCCV_aic**: 171.60ms (106.8x OLS), R²=0.4174, RMSE vs true=0.4178

### n=10, p=5

- **OLS**: 1.66ms (1.0x OLS), R²=0.8794, RMSE vs true=1.1057
- **PCR_constrained**: 25.53ms (15.3x OLS), R²=0.4058, RMSE vs true=0.1959
- **PCR_penalized**: 34.75ms (20.9x OLS), R²=0.4057, RMSE vs true=0.1939
- **PCCV_aic**: 612.50ms (368.3x OLS), R²=0.3985, RMSE vs true=0.1990

### n=10, p=10

- **OLS**: 1.77ms (1.0x OLS), R²=1.0000, RMSE vs true=2.1887
- **PCR_constrained**: 39.47ms (22.3x OLS), R²=0.7223, RMSE vs true=0.4736
- **PCR_penalized**: 43.47ms (24.5x OLS), R²=0.7223, RMSE vs true=0.4709
- **PCCV_aic**: 838.51ms (473.0x OLS), R²=0.7198, RMSE vs true=0.4470

### n=30, p=3

- **OLS**: 1.31ms (1.0x OLS), R²=0.3285, RMSE vs true=0.3052
- **PCR_constrained**: 5.49ms (4.2x OLS), R²=0.3285, RMSE vs true=0.3052
- **PCR_penalized**: 16.66ms (12.8x OLS), R²=0.3285, RMSE vs true=0.3046
- **PCCV_aic**: 222.97ms (170.8x OLS), R²=0.3282, RMSE vs true=0.2891

### n=30, p=5

- **OLS**: 1.04ms (1.0x OLS), R²=0.5291, RMSE vs true=0.5194
- **PCR_constrained**: 27.18ms (26.1x OLS), R²=0.5103, RMSE vs true=0.3432
- **PCR_penalized**: 30.05ms (28.8x OLS), R²=0.5103, RMSE vs true=0.3420
- **PCCV_aic**: 542.07ms (520.1x OLS), R²=0.5101, RMSE vs true=0.3321

### n=30, p=10

- **OLS**: 1.82ms (1.0x OLS), R²=0.8730, RMSE vs true=0.3343
- **PCR_constrained**: 41.24ms (22.6x OLS), R²=0.8452, RMSE vs true=0.1829
- **PCR_penalized**: 42.94ms (23.6x OLS), R²=0.8452, RMSE vs true=0.1820
- **PCCV_aic**: 962.27ms (528.4x OLS), R²=0.8449, RMSE vs true=0.1755

### n=50, p=3

- **OLS**: 1.53ms (1.0x OLS), R²=0.1836, RMSE vs true=0.1970
- **PCR_constrained**: 4.32ms (2.8x OLS), R²=0.1836, RMSE vs true=0.1970
- **PCR_penalized**: 14.98ms (9.8x OLS), R²=0.1836, RMSE vs true=0.1955
- **PCCV_aic**: 216.25ms (141.7x OLS), R²=0.1835, RMSE vs true=0.1837

### n=50, p=5

- **OLS**: 1.79ms (1.0x OLS), R²=0.4309, RMSE vs true=0.1796
- **PCR_constrained**: 17.82ms (10.0x OLS), R²=0.4309, RMSE vs true=0.1753
- **PCR_penalized**: 18.88ms (10.5x OLS), R²=0.4309, RMSE vs true=0.1750
- **PCCV_aic**: 339.52ms (189.6x OLS), R²=0.4306, RMSE vs true=0.1596

### n=50, p=10

- **OLS**: 0.96ms (1.0x OLS), R²=0.8727, RMSE vs true=0.2857
- **PCR_constrained**: 45.22ms (47.1x OLS), R²=0.8654, RMSE vs true=0.1866
- **PCR_penalized**: 48.70ms (50.7x OLS), R²=0.8654, RMSE vs true=0.1863
- **PCCV_aic**: 963.41ms (1003.8x OLS), R²=0.8654, RMSE vs true=0.1853

### n=100, p=3

- **OLS**: 1.30ms (1.0x OLS), R²=0.2871, RMSE vs true=0.1831
- **PCR_constrained**: 15.07ms (11.6x OLS), R²=0.2846, RMSE vs true=0.1544
- **PCR_penalized**: 16.98ms (13.0x OLS), R²=0.2846, RMSE vs true=0.1539
- **PCCV_aic**: 301.84ms (231.9x OLS), R²=0.2845, RMSE vs true=0.1505

### n=100, p=5

- **OLS**: 1.40ms (1.0x OLS), R²=0.5398, RMSE vs true=0.1443
- **PCR_constrained**: 4.54ms (3.3x OLS), R²=0.5398, RMSE vs true=0.1443
- **PCR_penalized**: 17.48ms (12.5x OLS), R²=0.5398, RMSE vs true=0.1434
- **PCCV_aic**: 259.45ms (185.9x OLS), R²=0.5397, RMSE vs true=0.1361

### n=100, p=10

- **OLS**: 1.72ms (1.0x OLS), R²=0.8482, RMSE vs true=0.3822
- **PCR_constrained**: 44.73ms (26.0x OLS), R²=0.8334, RMSE vs true=0.1985
- **PCR_penalized**: 56.25ms (32.7x OLS), R²=0.8334, RMSE vs true=0.1981
- **PCCV_aic**: 904.50ms (525.8x OLS), R²=0.8333, RMSE vs true=0.1952

## Coefficient Accuracy

PCRegression typically achieves **better coefficient accuracy** when:
- True coefficients satisfy the constraints (e.g., negative slopes)
- Sample size is small (n < 50)
- Features are correlated (multicollinearity)

The trade-off is **computation time**: PCRegression uses iterative optimization vs OLS's closed-form solution.

