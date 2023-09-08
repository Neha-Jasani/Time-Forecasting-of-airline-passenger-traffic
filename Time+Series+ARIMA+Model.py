#!/usr/bin/env python
# coding: utf-8

# # Time Series:

# # Exponential Smoothing Technique:

# ### Importing libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') 


# # Reading and Understanding Data:

# In[2]:


df_air = pd.read_csv('airline-passenger-traffic.csv',header=None)


# In[3]:


df_air.columns = ['Months','Passengers']


# In[4]:


df_air.head()


# In[5]:


df_air.shape


# In[6]:


df_air.describe()


# In[7]:


df_air.info()


# In[8]:


df_air.dtypes


# #### Converting into DateTime Series:

# In[9]:


df_air['Months'] = pd.to_datetime(df_air['Months'],format='%Y-%m')


# In[10]:


df_air = df_air.set_index('Months')


# In[11]:


df_air.head()


# ### Time Series Analysis:

# In[12]:


# plottig time series data:

df_air.plot(figsize=(10,5))
plt.title('Airline Passenger Traffic')
plt.show()


# ### Observation :-
# - Here in plot some data are missing.

# ###  Missing Value Treatment:
# - 1.Mean Imputation
# - 2.Linear Interpolation
# - 3.Using Last Observation carried forward

# In[13]:


# 1. Mean imputation
df_air_mean = pd.read_csv('airline-passenger-traffic.csv')
df_air_mean.columns = ['Months','Passengers']
df_air_mean['Months'] = df_air_mean['Months'].astype('datetime64[ns]')
df_air_mean.set_index('Months',inplace=True)


# In[14]:


df_air_mean['Passengers'].fillna(df_air_mean['Passengers'].mean(),inplace=True)


# In[15]:


df_air_mean.isnull().sum()


# In[16]:


# Plot the graph
df_air_mean.plot(figsize=(12,7))
plt.xlabel('Months')
plt.ylabel('passengers')
plt.show()


# In[17]:


# 2. Using Linear interpolation
df_air['Passengers'].fillna(df_air['Passengers'].interpolate(method='linear'),inplace=True)


# In[18]:


# Plot the graph
df_air.plot(figsize=(12,7))
plt.xlabel('Months')
plt.ylabel('passengers')
plt.show()


# In[19]:


# 3. Using Last Observation carried forward
df_air_last = pd.read_csv('airline-passenger-traffic.csv')
df_air_last.columns = ['Months','Passengers']
df_air_last['Months'] = df_air_last['Months'].astype('datetime64[ns]')
df_air_last.set_index('Months',inplace=True)


# In[20]:


# Here in method we use pad, ffill, bfill
df_air_last['Passengers'].fillna(method = 'pad',inplace=True)


# In[21]:


# Plot the graph
df_air_last.plot(figsize=(12,7))
plt.xlabel('Months')
plt.ylabel('passengers')
plt.show()


# ### Observation :-
# - Here we use LInear interpolation for missing value treatment bcz here in data trend is present.

# # Handling Outliers

# In[22]:


fig = plt.subplots(figsize=(12,2))
ax = sns.boxplot(df_air['Passengers'], whis=1.5)


# In[23]:


# Histogram plot

df_air['Passengers'].hist(figsize=(12,4))


# # Time Series Decomposition 

# ### Additive decomposition

# In[24]:


from statsmodels.api import tsa


# In[25]:


plt.figure(figsize=(12,8))
decomposiotion_add = tsa.seasonal_decompose(df_air.Passengers , model='additive') # additive seasonal index
decomposiotion_add.plot()
plt.show()


# ### Multiplicative Seasonal Decomposition

# In[26]:


decomposiotion_mul = tsa.seasonal_decompose(df_air.Passengers , model='multiplicative') # multiplicative seasonal index
decomposiotion_mul.plot()
plt.show()


# ### Observation :-
# - From Residual we can say that the series is Multiplicative.
# - Here in series Trens is present.

# # Augmented Dickey-Fuller (ADF) test :-

# In[27]:


from statsmodels.tsa.stattools import adfuller


# In[28]:


adf_test = adfuller(df_air['Passengers'])


# In[29]:


print('ADF Statistic : %f' % adf_test[0])
print('p-value: %f' % adf_test[1])


# ### Observation :-
# - p-value is < 0.05 then the series is non Stationary

# #  Kwiatkowski Phillips Schmidt Shin (KPSS) test :- 

# In[30]:


from statsmodels.tsa.stattools import kpss


# In[31]:


kpss_test = kpss(df_air['Passengers'])


# In[32]:


print('KPSS Statistic : %f' % kpss_test[0])
print('p-value: %f' % kpss_test[1])


# ### Observation :-
# - p-value is < 0.05 then the series is non Stationary

# In[33]:


# To make Series Stationary:-


# ## Box Cox Transformation

# In[34]:


from scipy.stats import boxcox


# - box cox transformation to make variance constant --> as series is multiplicative

# In[35]:


# boxcox transformation
df_air['boxcox'] = boxcox(df_air['Passengers'],lmbda=0)


# In[36]:


df_air.index = df_air.index


# In[37]:


df_air.head()


# In[38]:


plt.figure(figsize=(12,4))
plt.plot(df_air['boxcox'],label = 'After Box Cox transformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
plt.show()


# In[39]:


# checking for stationary on transformed series
adf_test = adfuller(df_air['boxcox'])


# In[40]:


print('p-value: %f' % adf_test[1])


# #### Observation :-
# - series is not stationay.

# ## Differencing

# In[41]:


df_air['boxcox_diff'] = df_air['boxcox']-df_air['boxcox'].shift()


# In[42]:


df_air.head()


# In[43]:


# dropping the null value
df_air.dropna(inplace=True)


# In[44]:


df_air.head()


# In[45]:


# check for stationary on transformed and differenced series
adf_test = adfuller(df_air['boxcox_diff'])


# In[46]:


print('p-value: %f' % adf_test[1])


# #### Observation :-
# - p value < 0.05 series is stationary.

# In[47]:


plt.figure(figsize=(12,4))
plt.plot(df_air['boxcox_diff'],label = 'After Box Cox transformation')
plt.legend(loc='best')
plt.title('After Box Cox transform')
plt.show()


# # Autocorrelation function (ACF)

# In[48]:


from statsmodels.graphics.tsaplots import plot_acf


# In[49]:


plt.figure(figsize=(12,4))
plot_acf(df_air['boxcox_diff'],ax=plt.gca(),lags=30)
plt.show()


# # Partial autocorrelation function (PACF)

# In[50]:


from statsmodels.graphics.tsaplots import plot_pacf


# In[51]:


plt.figure(figsize=(12,4))
plot_pacf(df_air['boxcox_diff'],ax=plt.gca(),lags=30)
plt.show()


# ### Split data into training and test set 

# In[52]:


train_len = 120
train = df_air['boxcox_diff'][0:train_len] # frist 120 months as training set
test = df_air['boxcox_diff'][train_len:] # remaining 24 months as test set


# In[53]:


train


# In[54]:


test


# # Auto regression method (AR)

# In[55]:


from statsmodels.tsa.arima_model import ARIMA


# In[56]:


model_ar = ARIMA(train,order=(1,0,0)) # 1 for AR , 0 for I , 0 for MA


# In[57]:


model_ar = model_ar.fit()


# In[58]:


print(model_ar.params)


# In[59]:


y_pred_ar = model_ar.predict(test.index.min(),test.index.max())


# In[60]:


test.index.min()


# In[61]:


y_pred_ar


# In[62]:


# Plot train,test and forecast
plt.figure(figsize=(12,4))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(y_pred_ar,label='AR method')
plt.legend(loc='best')
plt.title('AR method')
plt.show()


# #### Calculate RMSE and MAPE

# In[63]:


from sklearn.metrics import mean_squared_error


# In[64]:


rmse = np.sqrt(mean_squared_error(test,y_pred_ar)).round(2)
print('RMSE :',rmse)


# In[65]:


mape = np.round(np.mean(np.abs(test-y_pred_ar)/test)*100,2)
print('MAPE :',mape)


# # Moving Average method (MA)

# In[66]:


model_ma = ARIMA(train,order=(0,0,1)) # 0 for AR , 0 for I , 1 for MA


# In[67]:


model_ma = model_ma.fit()


# In[68]:


print(model_ma.params)


# In[69]:


y_pred_ma = model_ma.predict(test.index.min(),test.index.max())


# In[70]:


y_pred_ma


# In[71]:


# Plot train,test and forecast
plt.figure(figsize=(12,4))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(y_pred_ma,label='MA method')
plt.legend(loc='best')
plt.title('MA method')
plt.show()


# #### Calculate RMSE and MAPE

# In[72]:


rmse = np.sqrt(mean_squared_error(test,y_pred_ma)).round(2)
mape = np.round(np.mean(np.abs(test-y_pred_ma)/test)*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# # Auto regressive Moving average model (ARMA)

# In[73]:


model_ar_ma = ARIMA(train,order=(1,0,1)) # 0 for AR , 0 for I , 1 for MA


# In[74]:


model_ar_ma = model_ar_ma.fit()


# In[75]:


print(model_ar_ma.params)


# In[76]:


y_pred_arma = model_ar_ma.predict(test.index.min(),test.index.max())


# In[77]:


y_pred_arma


# In[78]:


# Plot train,test and forecast
plt.figure(figsize=(12,4))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(y_pred_arma,label='ARMA method')
plt.legend(loc='best')
plt.title('ARMA method')
plt.show()


# #### Calculate RMSE and MAPE

# In[79]:


rmse = np.sqrt(mean_squared_error(test,y_pred_arma)).round(2)
mape = np.round(np.mean(np.abs(test-y_pred_arma)/test)*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# # Autoregressive Integrated Moving Average (ARIMA)

# In[80]:


model_arima = ARIMA(train,order=(1,1,1)) # 0 for AR , 0 for I , 1 for MA


# In[81]:


model_arima = model_arima.fit()


# In[82]:


print(model_arima.params)


# In[83]:


y_pred_arima = model_arima.predict(test.index.min(),test.index.max())


# In[84]:


# Plot train,test and forecast
plt.figure(figsize=(12,4))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(y_pred_arima,label='ARIMA method')
plt.legend(loc='best')
plt.title('ARIMA method')
plt.show()


# #### Calculate RMSE and MAPE

# In[85]:


rmse = np.sqrt(mean_squared_error(test,y_pred_arima)).round(2)
mape = np.round(np.mean(np.abs(test-y_pred_arima)/test)*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# # SARIMA:

# In[86]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[87]:


model_sarima = SARIMAX(train,order=(1,1,1),seasonal_order=(1,1,1,12))


# In[88]:


model_sarima = model_sarima.fit()


# In[89]:


print(model_sarima.params)


# In[90]:


y_pred_sarima = model_sarima.predict(test.index.min(),test.index.max())


# In[91]:


# Plot train,test and forecast
plt.figure(figsize=(12,4))
plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(y_pred_sarima,label='ARMA method')
plt.legend(loc='best')
plt.title('ARMA method')
plt.show()


# #### Calculate RMSE and MAPE

# In[92]:


rmse = np.sqrt(mean_squared_error(test,y_pred_sarima)).round(2)
mape = np.round(np.mean(np.abs(test-y_pred_sarima)/test)*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)

