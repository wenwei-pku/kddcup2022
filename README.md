# kddcup2022
A solution to kddcup2022: Long-Short Term Forecasting for Active Power of a Wind Farm

## Implementation
The task is to forecast 10 minutely wind power of 134 turbines from a wind farm for the next 48 hours, given the relative locations and internal status. We break the task to the nowcasting (0-3h) and short term (3h - 48h) part, targeting a more precise recent forecasting utilizing the inertia of wind and mean prediction in longer forecast horizons respectively.

You can train the long-term model and short-term model by:
```bash
python run.py 
python run_lgb_dense.py 
```
repectively.