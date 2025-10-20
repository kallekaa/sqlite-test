-- Additional columns to support richer technical feature set.
ALTER TABLE features ADD COLUMN log_return_1d REAL;
ALTER TABLE features ADD COLUMN log_return_5d REAL;
ALTER TABLE features ADD COLUMN log_return_10d REAL;
ALTER TABLE features ADD COLUMN rolling_std_close_20d REAL;
ALTER TABLE features ADD COLUMN bollinger_upper_20d REAL;
ALTER TABLE features ADD COLUMN bollinger_lower_20d REAL;
ALTER TABLE features ADD COLUMN bollinger_bandwidth_20d REAL;
ALTER TABLE features ADD COLUMN macd_line REAL;
ALTER TABLE features ADD COLUMN macd_signal REAL;
ALTER TABLE features ADD COLUMN macd_hist REAL;
ALTER TABLE features ADD COLUMN rolling_sharpe_5d REAL;
ALTER TABLE features ADD COLUMN drawdown REAL;
ALTER TABLE features ADD COLUMN max_drawdown_20d REAL;
ALTER TABLE features ADD COLUMN streak_up INTEGER;
ALTER TABLE features ADD COLUMN streak_down INTEGER;
