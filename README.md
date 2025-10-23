# Stock Ingestor

Stock Ingestor is a small end-to-end pipeline that downloads daily equity data with `yfinance`, stores it in SQLite, and explores the results with Python tooling. It includes feature engineering scripts, Scikit-Learn baselines for regression/classification, and Keras-based neural networks for next-day predictions. Reports, trained models, and plots are written to the `reports/` directory so you can inspect outcomes after each run.
