
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

Project Overview
This project implements and evaluates advanced deep learning models for multi-step time series forecasting using attention mechanisms. The objective is to compare a baseline LSTM model with an Attention-based LSTM model on a complex synthetic time series dataset containing trend, multiple seasonalities, and noise.

Features
- Synthetic dataset generation with trend, seasonality, and noise
- Data preprocessing (scaling and windowing)
- Baseline LSTM forecasting model
- Attention-based LSTM forecasting model
- Performance evaluation using RMSE and MAE
- Attention weight visualization and interpretation

Project Structure
- attention_time_series_forecasting.py  -> Main executable script
- README.docx                           -> Project documentation

Requirements
- Python 3.8+
- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- python-docx

Installation
pip install tensorflow numpy pandas scikit-learn matplotlib python-docx

How to Run
python AITimeSeriesForecasting.py

Output
- Training logs for baseline and attention models
- RMSE and MAE performance comparison
- Attention weights visualization plot

Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Attention Mechanism
The attention layer learns which past time steps are most important for predicting future values, improving long-range dependency modeling.

Author
Sneha Sekar

License
For academic us
