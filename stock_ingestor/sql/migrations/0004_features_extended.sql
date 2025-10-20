-- Extend the features table with additional engineered metrics.
ALTER TABLE features ADD COLUMN return_1d REAL;
ALTER TABLE features ADD COLUMN return_5d REAL;
ALTER TABLE features ADD COLUMN return_10d REAL;
ALTER TABLE features ADD COLUMN rolling_mean_return_5d REAL;
ALTER TABLE features ADD COLUMN rolling_std_return_5d REAL;
ALTER TABLE features ADD COLUMN momentum_5d REAL;
ALTER TABLE features ADD COLUMN price_change REAL;

ALTER TABLE features ADD COLUMN sma_5 REAL;
ALTER TABLE features ADD COLUMN sma_10 REAL;
ALTER TABLE features ADD COLUMN sma_20 REAL;
ALTER TABLE features ADD COLUMN ema_5 REAL;
ALTER TABLE features ADD COLUMN ema_10 REAL;
ALTER TABLE features ADD COLUMN ema_20 REAL;
ALTER TABLE features ADD COLUMN close_to_sma_5 REAL;
ALTER TABLE features ADD COLUMN sma_ratio_5_20 REAL;

ALTER TABLE features ADD COLUMN rolling_std_close_5d REAL;
ALTER TABLE features ADD COLUMN rolling_std_close_10d REAL;
ALTER TABLE features ADD COLUMN true_range REAL;
ALTER TABLE features ADD COLUMN avg_true_range_5d REAL;
ALTER TABLE features ADD COLUMN volatility_ratio REAL;

ALTER TABLE features ADD COLUMN volume_change REAL;
ALTER TABLE features ADD COLUMN volume_sma_5 REAL;
ALTER TABLE features ADD COLUMN volume_sma_10 REAL;
ALTER TABLE features ADD COLUMN volume_zscore_5d REAL;
ALTER TABLE features ADD COLUMN price_volume_trend REAL;

ALTER TABLE features ADD COLUMN high_low_spread REAL;
ALTER TABLE features ADD COLUMN close_position_in_range REAL;
ALTER TABLE features ADD COLUMN rolling_high_20d REAL;
ALTER TABLE features ADD COLUMN rolling_low_20d REAL;
ALTER TABLE features ADD COLUMN close_to_rolling_high_20d REAL;
ALTER TABLE features ADD COLUMN rsi_14 REAL;
