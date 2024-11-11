import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import streamlit as st
import yfinance as yf
# from ut import pdict
from keras.src.saving import load_model

model = load_model('Stock Predictions Model.keras')
st.header("Stock Marker Price Prediction")

stock = st.text_input("Enter stock symbol",'')
start = '2012-02-01'
end = '2022-12-31'

data = yf.download(stock, start, end)
st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

pas_100days = data_train.tail(100)
data_test = pd.concat([pas_100days, data_test], ignore_index = True)
data_test_scaler = scaler.fit_transform(data_test)

st.subheader('Price vs Moving_avg_50')
MA_50days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(MA_50days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs Moving_avg_50 vs 100')
MA_100days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(MA_50days, 'r')
plt.plot(MA_100days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs Moving_avg_100 vs 200')
MA_200days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(MA_100days, 'r')
plt.plot(MA_200days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Prize vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label = 'Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
