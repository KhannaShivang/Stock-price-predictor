import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID","GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

stock_data = yf.download(stock,start,end)
model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock data")
st.write(stock_data)

train_data = int(len(stock_data)*0.7)
x_test = pd.dataFrame(stock_data.Close[train_data:])

def plot_graph(figsize , values, full_data):
    fig = plt.figures(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close,'b')
    return fig

st.subheader('Origional Close Price and MA for 100 days')
stock_data['MA_for_100_days']= stock_data.Close.rolling(100).mean
st.pyplot(plot_graph((15,6),stock_data['MA_for_100_days'],stock_data))

from sklearn.preprocessing import minmax_scale
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])
x_data=[]
y_data=[]
for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data=np.array(x_data)
y_data=np.array(y_data)

pridictions = model.predict(x_test)
inv_pre = scaler.inverse_transform(pridictions) 
inv_y_test=scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        'Origional_text_data':inv_y_test.reshape(-1),
        'Predicted_data':inv_pre.reshape(-1)
    } , 
    index=stock_data.index[train_data+100:len(stock_data)]
)

st.subheader("Origional values vs Pridicted values")
st.write(ploting_data)

st.subheader("origional Close Price vs Pridicted Close Price")
fig = plt.figures(figsize=(15,6))
plt.plot(pd.concat([stock_data.Close[:train_data+100],ploting_data] ,axis=0))
plt.legend(["Data - not used", "Origional Test Data" , "Pridected Test Data"])
st.pyplot(fig)