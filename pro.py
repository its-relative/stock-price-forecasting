#!/usr/bin/env python
# coding: utf-8

# ### Importing the libraries

# In[1]:


import pandas as pd
import numpy as np


# ### Lets import the Data

# In[2]:


data = pd.read_csv("RS.csv")
df = data.copy()
df


# ### Let's understand the data

# In[3]:


df.shape


# checking for null values

# In[4]:


df.isna().sum()


# In[5]:


df.dtypes 


# In[6]:


df["Date"] = pd.to_datetime(df["Date"])
df.set_index('Date', inplace= True)


# In[7]:


df.dtypes


# In[8]:


# date_col = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
# cat_cols = [col for col in df.columns if df[col].dtype == "object"]
con_cols = [col for col in df.columns if df[col].dtype == "int64" or df[col].dtype == "float64"]


# In[9]:


# cat_cols


# In[10]:


# date_col


# In[11]:


con_cols


# ### Lets visualize the data

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


sns.pairplot(df[con_cols], kind= 'reg', diag_kind='kde')


# In[14]:


df.columns[:-1]


# In[15]:


for i, col in enumerate(df.columns[1:]):
    plt.hist(df[col],alpha = 0.5, bins="auto")
    plt.title(col)
    plt.show()


# In[16]:


df.describe()


# In[17]:


df.info()


# In[18]:


df[con_cols].corr()


# In[19]:


heatmap = sns.heatmap(df[con_cols[:-1]].corr(), annot=True, cmap='coolwarm', fmt='.6f')
plt.title('Correlation Heatmap with Values')
plt.show()


# In[20]:


for i, col in enumerate(df[con_cols].columns):
    plt.subplot(1, len(df[con_cols].columns), i+1)
    plt.boxplot(df[col])
    plt.title(col)
    # plt.show()
plt.tight_layout()
plt.show()


# In[21]:


import mplfinance as mpf


# In[22]:


mpf.plot(df, type='candle', ylabel='Price', volume=True, style='charles')


# In[23]:


# Plotting the time series
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Time Series Plot of Close Price')
plt.legend()
plt.grid(True)
plt.show()


# In[24]:


# Plotting the volume-price relationship scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Volume'], df['Close'], label='Volume vs. Close', color='green', alpha=0.5)
plt.xlabel('Volume')
plt.ylabel('Close Price')
plt.title('Volume-Price Relationship Scatter Plot')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


# Calculate moving averages (e.g., 7-day, 30-day)
df['Close_7day_MA'] = df['Close'].rolling(window=7).mean()
df['Close_30day_MA'] = df['Close'].rolling(window=30).mean()

# Plotting the original data and moving averages
plt.figure(figsize=(12, 8))
plt.plot(df.index, df['Close'], label='Original Close', color='blue')
plt.plot(df.index, df['Close_7day_MA'], label='7-day Moving Average', color='red')
plt.plot(df.index, df['Close_30day_MA'], label='30-day Moving Average', color='green')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price with Moving Averages')
plt.legend()
plt.grid(True)
plt.show()



# In[26]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Autocorrelation Plot (ACF)
plt.figure(figsize=(12, 6))
plot_acf(df['Close'], lags=60, alpha=0.05) 
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.grid(True)
plt.show()

# Partial Autocorrelation Plot (PACF)
plt.figure(figsize=(12, 6))
plot_pacf(df['Close'], lags=60, alpha=0.05) 
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.grid(True)
plt.show()


# In[27]:


# Lagged Scatter Plots
plt.figure(figsize=(12, 6))
plt.scatter(df['Close'].shift(1), df['Close'], label='Close(t-1) vs. Close(t)', color='blue', alpha=0.5)
plt.xlabel('Close(t-1)')
plt.ylabel('Close(t)')
plt.title('Lagged Scatter Plot')
plt.legend()
plt.grid(True)
plt.show()



# # Building Models

# Predicting the closing stock price of Reliance

# In[28]:


import yfinance as yf
from datetime import datetime
ticker = "RS"
start = "2000-03-01"
end = datetime.now()
df = yf.download(tickers=ticker, start= start, end= end)


# In[29]:


# here we are printing shape of data
df.shape


# In[30]:


df


# Create a new data frame with only the closing price and convert it to an array. Here we are taking about 80% of the data as the training data.

# In[31]:


import math


# In[32]:


#Creating a new dataframe with only the 'Close' column
data = df.filter(['Close'])
#Converting the dataframe to a numpy array
dataset = data.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)
training_data_len # we will need the var for creating x_train, x_test and all.


# Now let's scale the data set to be values between 0 and 1 inclusive.

# In[33]:


from sklearn.preprocessing import MinMaxScaler


# In[34]:


# Lets Scale all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_train = scaler.fit_transform(dataset[:training_data_len])
scaled_data_test = scaler.transform(dataset[training_data_len:])


# In[35]:


scaled_data_train.shape


# In[36]:


scaled_data_test.shape


# In[37]:


# scaled_data_train[0:training_data_len].shape


# In[38]:


#Creating the scaled training data set
train_data = scaled_data_train[0:training_data_len  , : ]

#Spliting the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<= 61:
        print(x_train, "Train")
        print(y_train,"Test")
        print()


# In[39]:


#Lets Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[40]:


x_train.shape


# In[41]:


y_train.shape


# In[42]:


# Lets reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[43]:


from keras.models import Sequential
from keras.layers import Dense,LSTM


# In[44]:


#Lets Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


# In[45]:


# Lets Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[46]:


# Lets train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[47]:


scaled_data_train[-60:,:]


# In[48]:


# scaled test data with 60 days of scaled training data
# Lets create testing data set
test_data = np.concatenate((scaled_data_train[-60:,:],(scaled_data_test)),axis = 0)
test_data


# In[49]:


test_data.shape


# In[50]:


#Creating the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] 
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[51]:


# Lets convert x_test to a numpy array  
x_test = np.array(x_test)


# In[52]:


# Lets reshape the data into the shape accepted by the LSTM  
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# In[53]:


# now we are getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling


# In[54]:


# Lets calculate the value of RMSE 
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[55]:


#Lets Plot the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# In[56]:


print(valid)


# ### Evaluation

# Lets calculate the r2 score of our model for evaluation

# In[57]:


from sklearn.metrics import r2_score

r2 = r2_score(valid["Close"], valid["Predictions"])


# In[58]:


prevR2 = 0.9937106223816641


# In[59]:


r2


# In[60]:


x_test


# In[61]:


x_input = test_data[-100:]

len(x_input)


# for 365 days

# In[62]:


x_input = test_data[-60:]

len(x_input)


# In[63]:


x_input = x_input.reshape(1,-1)


# In[64]:


x_input.shape


# In[65]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[66]:


temp_input


# In[67]:


# demonstrate prediction for next 365 days
from numpy import array

lst_output=[]
n_steps=60  # 60 time steps
i=0
while(i<365):  # predicting for 365 days
    
    if(len(temp_input)>n_steps):  # condition based on n_steps
        x_input=np.array(temp_input[1:])[:n_steps]  # Slicing to ensure correct size
        if i <= 5:
            print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape((1, n_steps, 1))  # Reshape to match model input
        yhat = model.predict(x_input, verbose=0)
        if i <= 5:
            print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        if i >= 360:
            print(yhat[0])
        
        temp_input.extend(yhat[0].tolist())
        if i> 360:    
            print(len(temp_input))
        
        lst_output.extend(yhat.tolist())
        i=i+1

print(lst_output)


# In[68]:


from datetime import datetime, timedelta

# Extracting date ranges for test data and prediction data
date_range_test = df.iloc[-60:,:].index


# In[69]:


len(date_range_test)


# In[70]:


from datetime import datetime, timedelta
import pandas as pd
import holidays

# Get today's date
today = datetime.now()

# Calculate the date one year from now
one_year_from_now = today + timedelta(days=365+365-198)

# Initialize a holidays object for India
indian_holidays = holidays.India()

# Create a date range from today to one year from now, excluding weekends and holidays
date_range = pd.date_range(start=today, end=one_year_from_now, freq='B')

# Filter out the holidays and weekends from the date range
filtered_date_range = [
    date for date in date_range 
    if date not in indian_holidays and date.weekday() < 5  # Exclude holidays and weekdays (Monday to Friday)
]

# Convert the filtered date range to a pandas DateTimeIndex
filtered_date_range_index_pred = pd.DatetimeIndex(filtered_date_range)

# Print the filtered date range
print(filtered_date_range_index_pred)


# In[71]:


len(filtered_date_range_index_pred)


# In[72]:


day_new=np.arange(1,61)
day_pred=np.arange(61,61+365)


# In[73]:


import matplotlib.pyplot as plt


# In[74]:


len(data)


# In[75]:


lst_output_withoutScale = scaler.inverse_transform(lst_output)


# In[76]:


lst_output_withoutScale


# In[77]:


last_60_dayOut = scaler.inverse_transform(scaled_data_test[-60:])
print(last_60_dayOut)


# In[78]:


last_60_dayOut.shape


# In[79]:


df.iloc[-1:,:].index


# In[80]:


# Extract the relevant data for plotting
scaled_data_test_plot = scaler.inverse_transform(scaled_data_test[-60:])
lst_output_plot = scaler.inverse_transform(lst_output)

# Make sure date_range_test and filtered_date_range_index_pred have the same length
date_range_test = date_range_test[-60:]
filtered_date_range_index_pred = filtered_date_range_index_pred[:len(lst_output_plot)]

# Plot the data
plt.figure(figsize= (14,10))
plt.plot(date_range_test, scaled_data_test_plot, label='Scaled Data Test')
plt.plot(filtered_date_range_index_pred, lst_output_plot, label='Predicted Data')
plt.xlabel("Date")
plt.ylabel("Price Data")
plt.legend()
plt.show()


# In[81]:


scaled_data = np.concatenate((scaled_data_train, scaled_data_test),axis = 0)
df3=scaled_data.tolist()
df3.extend(lst_output)
plt.plot(df3[-200:])
plt.show()


# In[82]:


df3=scaler.inverse_transform(df3).tolist()


# In[83]:


full_date_index = df.index

full_date_index  = full_date_index.tolist()

full_date_index.extend(filtered_date_range_index_pred)

len(full_date_index)


# In[87]:


df4 = pd.DataFrame(df3,index= full_date_index)


# In[89]:


df4


# In[84]:


plt.plot(full_date_index, df3, label = "Date")
plt.ylabel("Price in ₹")
plt.show()


# In[91]:


r2


# In[92]:


import pickle
import os

name1 = str(r2)[:6]
name2 = "lstmforecast.pkl"

os.mkdir("./DataFrames")

full_name = name1+name2
# full_name
directory_model = './trained models/'
directory_dataFrame = './DataFrames/'
isfile = False

try:
    with open(f'{directory_model}{full_name}','r') as f:
        isfile = True
except Exception:
    pass
if isfile:
    new_rand = str(np.random.randint(1,10))
    new_name = new_rand+full_name
    with open(f'{directory_model}{new_name}', 'wb') as f:
        pickle.dump(model,f)
    with open(f'{directory_dataFrame}{new_name}','wb') as df:
        pickle.dump(df4,df)
else:
    with open(f"{directory_model}{full_name}","wb") as f:
        pickle.dump(model,f)
    with open(f'{directory_dataFrame}{full_name}','wb') as df:
        pickle.dump(df4,df)


# In[96]:


df4.tail()


# In[112]:


price = df4.loc["2024-07-18"]

price = price.values


# In[113]:


price


# In[ ]:




