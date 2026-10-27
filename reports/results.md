# Results

Auto-generated. Do not edit by hand.

## Signal IC

Spearman IC between the ranked signal at close t and the forward return from close t+1 to close t+1+h. t-stat uses every h-th date only (non-overlapping). Dates with fewer than 50 stocks are skipped.

| name | mean_ic | ic_std | ic_ir | pct_positive | t_stat | n_dates | n_non_overlap |
|---|---|---|---|---|---|---|---|
| equal_weight h=5 | 0.0096 | 0.2030 | 0.0471 | 0.5374 | 1.9715 | 3971 | 795 |
| equal_weight h=21 | -0.0011 | 0.1923 | -0.0059 | 0.5252 | 0.2096 | 3955 | 189 |
| equal_weight h=63 | -0.0169 | 0.1841 | -0.0916 | 0.4787 | -1.3327 | 3913 | 63 |
| low_vol h=5 | -0.0126 | 0.2491 | -0.0506 | 0.4691 | -1.5013 | 4163 | 833 |
| low_vol h=21 | -0.0286 | 0.2363 | -0.1209 | 0.4589 | -1.6475 | 4147 | 198 |
| low_vol h=63 | -0.0549 | 0.2379 | -0.2306 | 0.4037 | -1.9253 | 4105 | 66 |
| momentum h=5 | 0.0147 | 0.2179 | 0.0675 | 0.5424 | 2.1522 | 3971 | 795 |
| momentum h=21 | 0.0104 | 0.1953 | 0.0531 | 0.5532 | 0.9325 | 3955 | 189 |
| momentum h=63 | 0.0051 | 0.1771 | 0.0288 | 0.5229 | 0.0873 | 3913 | 63 |
| random h=5 | 0.0000 | 0.0464 | 0.0007 | 0.4966 | -1.7655 | 4223 | 845 |
| random h=21 | 0.0001 | 0.0462 | 0.0030 | 0.4977 | 2.2791 | 4207 | 201 |
| random h=63 | -0.0003 | 0.0456 | -0.0058 | 0.4936 | 1.9612 | 4165 | 67 |
| reversal h=5 | 0.0135 | 0.1684 | 0.0801 | 0.5242 | 2.0681 | 4218 | 844 |
| reversal h=21 | 0.0102 | 0.1539 | 0.0665 | 0.5278 | 0.3243 | 4202 | 201 |
| reversal h=63 | 0.0097 | 0.1466 | 0.0662 | 0.5149 | 0.4639 | 4160 | 67 |

## Quintile forward returns (21d)

Average 21-day forward return per signal quintile (Q1 = lowest signal), averaged over all dates.

| name | Q1 | Q2 | Q3 | Q4 | Q5 | Q5-Q1 |
|---|---|---|---|---|---|---|
| momentum | 0.0155 | 0.0131 | 0.0134 | 0.0130 | 0.0174 | 0.0019 |
| reversal | 0.0154 | 0.0137 | 0.0139 | 0.0146 | 0.0178 | 0.0023 |
| low_vol | 0.0230 | 0.0144 | 0.0134 | 0.0117 | 0.0107 | -0.0124 |
| random | 0.0150 | 0.0151 | 0.0150 | 0.0150 | 0.0149 | -0.0001 |
| equal_weight | 0.0169 | 0.0151 | 0.0133 | 0.0136 | 0.0135 | -0.0034 |

## Backtest: long-short (monthly)

Quintile portfolios, cost 10 bps per unit traded, exec_lag=1. Gross and net of costs. Short borrow costs ignored. Weight drift between rebalances ignored. Risk-free rate ignored.

| name | ann_return_gross | ann_return_net | ann_vol_net | sharpe_gross | sharpe_net | max_drawdown_net | calmar_net | ann_turnover | ann_cost | hit_rate_net | worst_month_net | breakeven_bps | ann_return_long | ann_return_short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum | 0.0227 | 0.0114 | 0.1804 | 0.1259 | 0.0634 | -0.5101 | 0.0224 | 11.2600 | 0.0113 | 0.5319 | -0.1350 | 20.1528 | 0.2120 | -0.1893 |
| reversal | 0.0084 | -0.0288 | 0.1287 | 0.0658 | -0.2239 | -0.5683 | -0.0507 | 37.2531 | 0.0373 | 0.4861 | -0.1832 | 2.2651 | 0.2080 | -0.1995 |
| low_vol | -0.1463 | -0.1562 | 0.2028 | -0.7213 | -0.7701 | -0.9538 | -0.1637 | 9.9130 | 0.0099 | 0.4641 | -0.1776 | 0.0000 | 0.1337 | -0.2800 |
| random | -0.0134 | -0.0518 | 0.0401 | -0.3450 | -1.2943 | -0.5973 | -0.0868 | 38.4121 | 0.0384 | 0.4747 | -0.0376 | 0.0000 | 0.1771 | -0.1905 |
| equal_weight | -0.0517 | -0.0778 | 0.1515 | -0.3420 | -0.5134 | -0.8085 | -0.0962 | 26.0577 | 0.0261 | 0.5043 | -0.1232 | 0.0000 | 0.1629 | -0.2146 |

## Backtest: long-only vs equal-weight (monthly)

Top quintile only. excess_* = long-only net minus the equal-weight benchmark net (all stocks with a price, same rebalance dates and costs).

| name | ann_return_gross | ann_return_net | ann_vol_net | sharpe_gross | sharpe_net | max_drawdown_net | calmar_net | ann_turnover | ann_cost | hit_rate_net | worst_month_net | breakeven_bps | ann_return_long | ann_return_short | bench_ann_return_net | excess_ann_return_net | excess_sharpe_net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum | 0.2120 | 0.2063 | 0.2062 | 1.0284 | 1.0007 | -0.3640 | 0.5667 | 5.6991 | 0.0057 | 0.5650 | -0.1275 | 371.9940 | 0.2120 | 0.0000 | 0.1670 | 0.0260 | 0.2781 |
| reversal | 0.2080 | 0.1894 | 0.2137 | 0.9732 | 0.8864 | -0.4995 | 0.3792 | 18.5405 | 0.0185 | 0.5507 | -0.2740 | 112.1767 | 0.2080 | 0.0000 | 0.1855 | 0.0030 | 0.0408 |
| low_vol | 0.1337 | 0.1283 | 0.1369 | 0.9770 | 0.9372 | -0.3347 | 0.3833 | 5.4482 | 0.0054 | 0.5589 | -0.1113 | 245.4483 | 0.1337 | 0.0000 | 0.1795 | -0.0531 | -0.5824 |
| random | 0.1771 | 0.1578 | 0.1805 | 0.9812 | 0.8744 | -0.4020 | 0.3925 | 19.2561 | 0.0193 | 0.5514 | -0.1930 | 91.9472 | 0.1771 | 0.0000 | 0.1855 | -0.0285 | -1.1643 |
| equal_weight | 0.1629 | 0.1491 | 0.1650 | 0.9874 | 0.9033 | -0.3849 | 0.3873 | 13.8319 | 0.0138 | 0.5576 | -0.1545 | 117.7624 | 0.1629 | 0.0000 | 0.1670 | -0.0276 | -0.4032 |

## Cost sensitivity: long-short net Sharpe (monthly)

| name | 0 bps | 5 bps | 10 bps | 20 bps | 50 bps |
|---|---|---|---|---|---|
| momentum | 0.1218 | 0.0915 | 0.0613 | 0.0009 | -0.1791 |
| reversal | 0.0656 | -0.0792 | -0.2234 | -0.5074 | -1.2821 |
| low_vol | -0.7160 | -0.7402 | -0.7644 | -0.8128 | -0.9563 |
| random | -0.3442 | -0.8323 | -1.2912 | -2.0504 | -3.1244 |
| equal_weight | -0.3308 | -0.4139 | -0.4966 | -0.6602 | -1.1276 |

## Backtest: long-short (weekly)

Quintile portfolios, cost 10 bps per unit traded, exec_lag=1. Gross and net of costs. Short borrow costs ignored. Weight drift between rebalances ignored. Risk-free rate ignored.

| name | ann_return_gross | ann_return_net | ann_vol_net | sharpe_gross | sharpe_net | max_drawdown_net | calmar_net | ann_turnover | ann_cost | hit_rate_net | worst_month_net | breakeven_bps | ann_return_long | ann_return_short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum | 0.0326 | 0.0090 | 0.1837 | 0.1775 | 0.0489 | -0.5590 | 0.0161 | 23.6278 | 0.0236 | 0.5326 | -0.1446 | 13.8017 | 0.2212 | -0.1886 |
| reversal | 0.0483 | -0.1125 | 0.1520 | 0.3185 | -0.7398 | -0.8816 | -0.1276 | 160.7737 | 0.1608 | 0.4606 | -0.1111 | 3.0054 | 0.2248 | -0.1765 |
| low_vol | -0.1476 | -0.1649 | 0.2064 | -0.7152 | -0.7989 | -0.9618 | -0.1715 | 17.3031 | 0.0173 | 0.4650 | -0.1616 | 0.0000 | 0.1273 | -0.2749 |
| random | 0.0085 | -0.1586 | 0.0438 | 0.2178 | -3.6186 | -0.9308 | -0.1704 | 167.1217 | 0.1671 | 0.4094 | -0.0730 | 0.5087 | 0.1778 | -0.1693 |
| equal_weight | -0.0193 | -0.1189 | 0.1572 | -0.1232 | -0.7565 | -0.9015 | -0.1319 | 99.5844 | 0.0996 | 0.4908 | -0.1378 | 0.0000 | 0.1752 | -0.1945 |

## Backtest: long-only vs equal-weight (weekly)

Top quintile only. excess_* = long-only net minus the equal-weight benchmark net (all stocks with a price, same rebalance dates and costs).

| name | ann_return_gross | ann_return_net | ann_vol_net | sharpe_gross | sharpe_net | max_drawdown_net | calmar_net | ann_turnover | ann_cost | hit_rate_net | worst_month_net | breakeven_bps | ann_return_long | ann_return_short | bench_ann_return_net | excess_ann_return_net | excess_sharpe_net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum | 0.2212 | 0.2092 | 0.2045 | 1.0817 | 1.0233 | -0.3639 | 0.5750 | 11.9731 | 0.0120 | 0.5704 | -0.1286 | 184.7453 | 0.2212 | 0.0000 | 0.1689 | 0.0276 | 0.2934 |
| reversal | 0.2248 | 0.1445 | 0.2205 | 1.0195 | 0.6555 | -0.4926 | 0.2934 | 80.3008 | 0.0803 | 0.5443 | -0.2329 | 27.9967 | 0.2248 | 0.0000 | 0.1827 | -0.0386 | -0.4532 |
| low_vol | 0.1273 | 0.1173 | 0.1346 | 0.9452 | 0.8713 | -0.3418 | 0.3431 | 9.9998 | 0.0100 | 0.5528 | -0.1423 | 127.2590 | 0.1273 | 0.0000 | 0.1788 | -0.0633 | -0.6853 |
| random | 0.1778 | 0.0942 | 0.1808 | 0.9838 | 0.5209 | -0.4085 | 0.2305 | 83.5888 | 0.0836 | 0.5390 | -0.2029 | 21.2661 | 0.1778 | 0.0000 | 0.1830 | -0.0889 | -3.2945 |
| equal_weight | 0.1752 | 0.1222 | 0.1663 | 1.0526 | 0.7346 | -0.3564 | 0.3428 | 52.9858 | 0.0530 | 0.5475 | -0.1061 | 33.0596 | 0.1752 | 0.0000 | 0.1689 | -0.0541 | -0.7579 |

## Cost sensitivity: long-short net Sharpe (weekly)

| name | 0 bps | 5 bps | 10 bps | 20 bps | 50 bps |
|---|---|---|---|---|---|
| momentum | 0.1720 | 0.1097 | 0.0474 | -0.0772 | -0.4499 |
| reversal | 0.3181 | -0.2114 | -0.7389 | -1.7616 | -4.2562 |
| low_vol | -0.7099 | -0.7514 | -0.7929 | -0.8759 | -1.1234 |
| random | 0.2176 | -1.8655 | -3.6159 | -5.7830 | -7.5446 |
| equal_weight | -0.1194 | -0.4268 | -0.7330 | -1.3362 | -2.9863 |

## Execution lag comparison (exec_lag=0 is OPTIMISTIC)

exec_lag=0 trades at the same close the signal uses. That is not tradable in practice, so these numbers are an optimistic upper bound for comparison only.

| name | sharpe_gross_lag1 | sharpe_gross_lag0_OPTIMISTIC | sharpe_net_lag1 | sharpe_net_lag0_OPTIMISTIC |
|---|---|---|---|---|
| momentum monthly | 0.1259 | 0.1524 | 0.0634 | 0.0902 |
| reversal monthly | 0.0658 | 0.0971 | -0.2239 | -0.1920 |
| low_vol monthly | -0.7213 | -0.7002 | -0.7701 | -0.7489 |
| random monthly | -0.3450 | -0.3460 | -1.2943 | -1.3004 |
| equal_weight monthly | -0.3420 | -0.2919 | -0.5134 | -0.4628 |
| momentum weekly | 0.1775 | 0.2030 | 0.0489 | 0.0744 |
| reversal weekly | 0.3185 | 0.5488 | -0.7398 | -0.4658 |
| low_vol weekly | -0.7152 | -0.6835 | -0.7989 | -0.7672 |
| random weekly | 0.2178 | 0.1580 | -3.6186 | -3.6660 |
| equal_weight weekly | -0.1232 | -0.0173 | -0.7565 | -0.6477 |

## Walk-forward (out-of-sample)

Out-of-sample only: 2015-01-02 to 2026-10-26, 12 test years. Every row covers the same dates. Long-short quintiles, monthly rebalance, 10 bps costs, exec_lag=1. Weights fit with method `ic_weighted` on 21-day forward returns.

| name | ann_return_gross | ann_return_net | ann_vol_net | sharpe_gross | sharpe_net | max_drawdown_net | calmar_net | ann_turnover | ann_cost | hit_rate_net | worst_month_net | breakeven_bps | ann_return_long | ann_return_short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walk_forward | -0.0009 | -0.0309 | 0.1542 | -0.0060 | -0.2006 | -0.5492 | -0.0563 | 30.0175 | 0.0300 | 0.5108 | -0.1878 | 0.0000 | 0.1960 | -0.1969 |
| momentum | 0.0199 | 0.0087 | 0.1999 | 0.0994 | 0.0438 | -0.5101 | 0.0172 | 11.1177 | 0.0111 | 0.5349 | -0.1350 | 17.8688 | 0.2092 | -0.1893 |
| reversal | 0.0045 | -0.0328 | 0.1456 | 0.0311 | -0.2251 | -0.5157 | -0.0636 | 37.2871 | 0.0373 | 0.4888 | -0.1832 | 1.2102 | 0.1940 | -0.1895 |
| low_vol | -0.1614 | -0.1718 | 0.2185 | -0.7388 | -0.7866 | -0.9214 | -0.1865 | 10.4498 | 0.0104 | 0.4668 | -0.1776 | 0.0000 | 0.1207 | -0.2821 |
| random | -0.0107 | -0.0491 | 0.0424 | -0.2591 | -1.1602 | -0.4637 | -0.1060 | 38.4687 | 0.0385 | 0.4776 | -0.0376 | 0.0000 | 0.1634 | -0.1741 |
| equal_weight | -0.0633 | -0.0892 | 0.1643 | -0.3862 | -0.5428 | -0.7560 | -0.1179 | 25.8389 | 0.0258 | 0.5153 | -0.1232 | 0.0000 | 0.1520 | -0.2153 |


## Walk-forward fitted weights

Weight per signal for each test year, fit on that window's training data only.

| name | momentum | reversal | low_vol |
|---|---|---|---|
| 2015 | 0.7694 | 0.2306 | 0.0000 |
| 2016 | 0.8611 | 0.1389 | 0.0000 |
| 2017 | 0.7123 | 0.2877 | 0.0000 |
| 2018 | 0.4068 | 0.5932 | 0.0000 |
| 2019 | 0.0000 | 1.0000 | 0.0000 |
| 2020 | 0.0000 | 1.0000 | 0.0000 |
| 2021 | 0.0000 | 1.0000 | 0.0000 |
| 2022 | 0.0000 | 1.0000 | 0.0000 |
| 2023 | 0.0000 | 1.0000 | 0.0000 |
| 2024 | 0.0000 | 1.0000 | 0.0000 |
| 2025 | 0.6580 | 0.3420 | 0.0000 |
| 2026 | 0.6631 | 0.3369 | 0.0000 |

