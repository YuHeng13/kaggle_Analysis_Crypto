# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:26:22 2024

@author: j_171
"""
#https://www.kaggle.com/code/mianbilal12/forcasting-bitcoin-with-arima-sarima-and-lstm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import os
from finlab_crypto import crawler


# 關閉警告
import warnings
warnings.filterwarnings("ignore")

# 設定 Matplotlib 與 Seaborn 的配置
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_style("whitegrid")

# Pandas 的顯示配置
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)  # Display all columns

#  載入數據
Doge = crawler.get_all_binance('BTCUSDT', '1d')
df = Doge

# 日期轉換
df['timestamp'] = pd.to_datetime(df.index, errors='coerce')  # Convert the 'Date' column to datetime. Invalid formats will become NaT.

    
# 驗證日期格式

if df['timestamp'].isnull().any():  # If there are null values in 'Date' after conversion:

    print("Warning: Some dates could not be parsed. Check the dataset for invalid date formats.")


df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

#提取年份
df['year'] = df['timestamp'].dt.year  # Extract the year from the 'Date' column and create a new 'year' column.
print(df.dtypes)

df = df.rename(columns={
    'timestamp' : 'Date','open' : 'Open',    'high' : 'High',    'low': 'Low', 'close' : 'Close' , 'volume' : 'Volume'    
    })

#計算交易值
#基於價格和交易量估算交易值
df['dollars'] = 0.5 * (df['High'] + df['Low']) * df['Volume']  
# 顯示數據
df.head()

df.shape

df.tail()

#檢查比特幣的最大收盤價

print((df['Close']).max())

print((df['Date']).max())


#檢查比特幣的最小收盤價
#結構檢查，了解數據集的完整性和整潔程度，為進一步的分析做好準備
print((df['Close']).min())

print((df['Date']).min())

df.info()

print("Summary Statistics:\n", df.describe())

'''Expolatory Data Analysis (EDA)'''

#箱型圖

plt.figure(figsize=(12, 6))

sns.boxplot(x='year', y='Close', data=df, color='#661d1c')

plt.title('Yearly Distribution of Bitcoin Prices', fontsize=16)

plt.xlabel('Year', fontsize=12)

plt.ylabel('Closing Price', fontsize=12)

plt.tight_layout()

plt.show()

# 年份分色的散點圖
from seaborn import scatterplot

plt.figure(figsize=(10, 5))

scatterplot(data=df, x='Date', y='Close', hue='year', color='#ab2222')
#sns.scatterplot(data=df, x='Date', y='Close', hue='year', palette='cool', alpha=0.7)


#比特幣折線圖
plt.figure(figsize=(12, 6))

sns.lineplot(data=df, x='Date', y='Close', label='Closing Price', color='#34ebab')

plt.title('Bitcoin Closing Price Over Time')

plt.xlabel('Date')

plt.ylabel('Price (USD)')

plt.legend()

plt.show()

#比特幣每日價格範圍(最高到最低)
plt.figure(figsize=(12, 6))

sns.lineplot(data=df, x='Date', y=df['High'] - df['Low'], label='Daily Range',color='#271d3b')

plt.title('Bitcoin Daily Price Range')

plt.xlabel('Date')

plt.ylabel('Price Difference (USD)')

plt.legend()

plt.show()



from seaborn import regplot


#比特幣的收盤價如何隨時間變化
plt.figure(figsize=(10, 5))

regplot(data=df, x='year', y='Close',color='#ab2222')

#比特幣交易量（Volume）的折線圖
plt.figure(figsize=(12, 6))

sns.lineplot(data=df, x='Date', y='Volume', label='Volume Traded',color='#4d22ab')

plt.title('Bitcoin Volume Traded Over Time')

plt.xlabel('Date')

plt.ylabel('Volume')

plt.legend()

plt.show()

#計算相關矩陣-皮爾森相關係數

corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')



# Plotting the histogram

plt.figure(figsize=(10, 6))
'''
bins=30：將數據分為 30 個區間，設定直方圖的柱數。
如果區間太少，可能會掩蓋數據細節。如果區間太多，可能會增加噪音。
kde=True：同時繪製核密度估計（Kernel Density Estimation）曲線，展示數據的平滑分佈。
'''
sns.histplot(df['Close'], bins=30, kde=True, color='skyblue')
#sns.histplot(df, x='Close', bins=30, kde=True, hue='year', palette='viridis')
#添加標題與軸標籤

plt.title('Distribution of Bitcoin Closing Prices', fontsize=16)

plt.xlabel('Closing Price', fontsize=12)

plt.ylabel('Frequency', fontsize=12)

#mean_close = df['Close'].mean()
#plt.axvline(mean_close, color='red', linestyle='--', label='Mean')

#顯示圖表

plt.tight_layout()

plt.show()

#數據預處理深度學習模型

#提取收盤價數據

data = df.filter(['Close'])

#將 DataFrame 轉為 NumPy 陣列

dataset = data.values

#計算訓練數據的長度

training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len


'''
validation_data_len = int(len(dataset) * .05)
training_data_len = len(dataset) - validation_data_len

data = df.filter(['Close']).dropna()  # 去除空值

print("Total rows:", len(dataset))
print("Training rows:", training_data_len)
'''

# 初始化和縮放數據

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)

scaled_data

# 建立訓練數據集

#建立縮放的訓練數據集

train_data = scaled_data[0:int(training_data_len), :]

#構建 x_train 和 y_train

x_train = []

y_train = []

#每次取 60 天的數據作為特徵，當前日期的數據作為標籤
        
for i in range(60, len(train_data)):

    x_train.append(train_data[i-60:i, 0])

    y_train.append(train_data[i, 0])

    if i<= 61:

        print(x_train)

        print(y_train)

        print()
      

#將數據拆分為 x_train 和 y_train 數據集

x_train, y_train = np.array(x_train), np.array(y_train)


# Reshape the data

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# x_train.shape

#LSTM Model
from keras.models import Sequential

from keras.layers import Dense, LSTM


# Build the LSTM model

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(25))

model.add(Dense(1))

# Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model

model.fit(x_train, y_train, batch_size=1, epochs=2)


# Create the testing data set

test_data = scaled_data[training_data_len - 60: , :]

# Create the data sets x_test and y_test

x_test = []

y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):

    x_test.append(test_data[i-60:i, 0])

    

# Convert the data to a numpy array

x_test = np.array(x_test)



# Reshape the data

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))



# Get the models predicted price values 

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)



# Get the root mean squared error (RMSE)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

rmse

# Plot the data

train = data[:training_data_len]

valid = data[training_data_len:]

valid['Predictions'] = predictions

# Visualize the data

plt.figure(figsize=(16,6))

plt.title('Model')

plt.xlabel('Date', fontsize=18)

plt.ylabel('Close Price USD ($)', fontsize=18)

plt.plot(train['Close'])

plt.plot(valid[['Close', 'Predictions']])

plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

plt.show()

#Check Stationarity 檢查平穩性
# stationarity check on data

from statsmodels.tsa.stattools import adfuller

def adf_test(df):

    result = adfuller(df)

    print('ADF Statistic: %f' % result[0])

    print('p-value: %f' % result[1])

    if result[1] <= 0.05:

        print("Reject the null hypothesis. Data is stationary")

    else:

        print("Fail to reject the null hypothesis. Data is not stationary")

adf_test(df['Close'])   

# Importing required libraries

from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt



# Decompose the 'Close' column to see trend, seasonality, and residuals

decompose = seasonal_decompose(df['Close'], model='additive', period=30)



# Plot with customized colors

plt.figure(figsize=(10, 8))



# Plot Trend Component

plt.subplot(411)

plt.plot(df['Date'], decompose.trend, color='orangered', label='Trend')

plt.legend(loc='best')

plt.title('Trend Component')



# Plot Seasonal Component

plt.subplot(412)

plt.plot(df['Date'], decompose.seasonal, color='mediumseagreen', label='Seasonality')

plt.legend(loc='best')

plt.title('Seasonal Component')



# Plot Residual Component

plt.subplot(413)

plt.plot(df['Date'], decompose.resid, color='dodgerblue', label='Residual')

plt.legend(loc='best')

plt.title('Residual Component')



# Plot Observed (Original) Data

plt.subplot(414)

plt.plot(df['Date'], df['Close'], color='darkgray', label='Observed')

plt.legend(loc='best')

plt.title('Observed Data')



# Adjust layout to prevent overlap

plt.tight_layout()

plt.show()

#ACF 和差分可視化
from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as plt

#------

# Set custom style for better visuals

#plt.style.use('seaborn-darkgrid')



# Create subplots for original series and differencing

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)



# Original Series

axes[0, 0].plot(df['Close'], color='royalblue')

axes[0, 0].set_title('Original Series', fontsize=14, color='darkred')

plot_acf(df['Close'], ax=axes[0, 1], lags=40, color='darkblue', title='ACF - Original')



# 1st Differencing

axes[1, 0].plot(df['Close'].diff(), color='orange')

axes[1, 0].set_title('1st Order Differencing', fontsize=14, color='darkgreen')

plot_acf(df['Close'].diff().dropna(), ax=axes[1, 1], lags=40, color='teal', title='ACF - 1st Diff')



# 2nd Differencing

axes[2, 0].plot(df['Close'].diff().diff(), color='purple')

axes[2, 0].set_title('2nd Order Differencing', fontsize=14, color='brown')

plot_acf(df['Close'].diff().diff().dropna(), ax=axes[2, 1], lags=40, color='magenta', title='ACF - 2nd Diff')



# Adjust layout and show the plot

plt.tight_layout()

plt.show()

#p、d 和q的含義
# plots

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# pd.plotting.autocorrelation_plot(df['Close'])



# plot_acf(df['Close'], alpha=0.05)



from statsmodels.tsa.stattools import acf, pacf

x_acf = pd.DataFrame(acf(df['Close']))

print(x_acf)

# partial autocorrelation

from statsmodels.tsa.stattools import acf, pacf

plot_pacf(df['Close'], lags=20, alpha=0.05)


#ARIMA Model
# Split the data into training and testing datasets for arima 

train_data = dataset[:training_data_len, 0]  # First 95% for training

test_data = dataset[training_data_len:, 0]   # Remaining 5% for testing



# Define ARIMA parameters (p, d, q)
from statsmodels.tsa.arima.model import ARIMA


p, d, q = 2, 1, 2  # You can tune these values for better results

# Create and fit the ARIMA model

arima_model = ARIMA(train_data, order=(p, d, q))

arima_fit = arima_model.fit()


from sklearn.metrics import mean_squared_error

# Forecast for the remaining 5% of data

forecast_steps = len(test_data)  # Number of steps equals the test data length

arima_forecast = arima_fit.forecast(steps=forecast_steps)



# Calculate RMSE for evaluation

rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))

print(f"ARIMA RMSE: {rmse}")



from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define SARIMA parameters



# (P, D, Q, s) for seasonal component (s = 12 for monthly seasonality, adjust as needed)

p, d, q = 2, 1, 2

P, D, Q, s = 1, 1, 1, 12  # Adjust `s` if there's no seasonality in your data



# Create and fit the SARIMA model

sarima_model = SARIMAX(train_data, 

                       order=(p, d, q), 

                       seasonal_order=(P, D, Q, s), 

                       enforce_stationarity=False, 

                       enforce_invertibility=False)

sarima_fit = sarima_model.fit(disp=False)



# Forecast for the remaining 5% of data

forecast_steps = len(test_data)  # Number of steps equals the test data length

sarima_forecast = sarima_fit.forecast(steps=forecast_steps)



# Calculate RMSE for evaluation

rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))

print(f"SARIMA RMSE: {rmse}")


sarima_forecast.max()




# Plot the data and SARIMA forecast

plt.figure(figsize=(12, 6))

plt.plot(range(len(train_data)), train_data, label='Training Data')

plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_data, label='Test Data')

plt.plot(range(len(train_data), len(train_data) + len(test_data)), sarima_forecast, label='SARIMA Forecast', color='red')

plt.title('SARIMA Model: Forecast vs Actual')

plt.xlabel('Time')

plt.ylabel('Close Price')

plt.legend()

plt.show()


# 確保日期範圍正確
train_dates = pd.date_range(start="2023-01-01", periods=len(train_data), freq="D")
test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=1), periods=len(test_data), freq="D")

# 將數據轉換為 Pandas Series
train_series = pd.Series(train_data, index=train_dates, name="Train Data")
test_series = pd.Series(test_data, index=test_dates, name="Test Data")

# 合併完整數據
full_series = pd.concat([train_series, test_series])

# 篩選最近一年的數據
cutoff_date = full_series.index.max() - pd.Timedelta(days=365)
latest_year_data = full_series[full_series.index >= cutoff_date]

# 獲取測試數據在最近一年中的部分
latest_test_series = latest_year_data[latest_year_data.index.isin(test_series.index)]

# 確保 SARIMA 預測與測試數據對齊
forecast_values = sarima_forecast[-len(latest_test_series):]

# 繪圖
plt.figure(figsize=(12, 6))
plt.plot(latest_year_data.index, latest_year_data, label="Latest Year Data")
plt.plot(latest_test_series.index, latest_test_series, label="Test Data")
plt.plot(latest_test_series.index, forecast_values, label="SARIMA Forecast", color="red")

plt.title("SARIMA Model: Forecast vs Actual (Last Year)")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.show()