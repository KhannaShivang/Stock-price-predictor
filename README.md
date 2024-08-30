# Stock Price Predictor

This project leverages 20 years of historical stock data to predict daily closing prices using machine learning. It includes an interactive web application built with Streamlit for easy data input, prediction, and visualization.

## Key Features

- **Data Processing:** Handles 20 years of stock data, cleans it, and engineers features like moving averages and RSI for improved prediction accuracy.
- **Machine Learning Models:** Utilizes models such as LSTM or GRU to predict daily closing prices based on historical trends.
- **Evaluation:** Measures model performance using metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
- **Web App:** The Streamlit-based web app provides an intuitive interface for users to input data, run predictions, and visualize results in real time.

## Technologies Used

- Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, and Streamlit.

## How to Use

1. Clone the repo and install dependencies.
2. Run data preprocessing and model training scripts.
3. Launch the Streamlit app with `streamlit run app.py` to start predicting and visualizing stock prices.

## Future Enhancements

- Real-time data integration.
- Expanded web app features, including portfolio tracking.
