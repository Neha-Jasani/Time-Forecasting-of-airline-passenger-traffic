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


df_air.head()


# In[4]:


df_air.columns = ['Months','Passengers']


# In[5]:


df_air.head()


# In[6]:


df_air.shape


# In[7]:


df_air.describe()


# In[8]:


df_air.info()


# In[9]:


df_air.dtypes


# #### Converting into DateTime Series:

# In[10]:


df_air['Months'] = pd.to_datetime(df_air['Months'],format='%Y-%m')


# In[11]:


df_air = df_air.set_index('Months')


# In[12]:


df_air.head()


# ### Time Series Analysis:

# In[13]:


# plottig time series data:

df_air.plot(figsize=(10,5))
plt.title('Airline Passenger Traffic')
plt.show()


# #### Observation :-
# - Looking at the graph above, we can say that it is missing in the graph.
# - also we can say that in time series data Trend and Seasonality is present.

# ###  Missing Value Treatment:
# - 1.Mean Imputation
# - 2.Linear Interpolation
# - 3.Using Last Observation carried forward

# In[14]:


# 1. Mean imputation
df_air_mean = pd.read_csv('airline-passenger-traffic.csv')
df_air_mean.columns = ['Months','Passengers']
df_air_mean['Months'] = df_air_mean['Months'].astype('datetime64[ns]')
df_air_mean.set_index('Months',inplace=True)


# In[15]:


df_air_mean['Passengers'].fillna(df_air_mean['Passengers'].mean(),inplace=True)


# In[16]:


df_air_mean.isnull().sum()


# In[17]:


# Plot the graph
df_air_mean.plot(figsize=(12,7))
plt.xlabel('Months')
plt.ylabel('passengers')
plt.show()


# In[18]:


# 2. Using Linear interpolation
df_air['Passengers'].fillna(df_air['Passengers'].interpolate(method='linear'),inplace=True)


# In[19]:


# Plot the graph
df_air.plot(figsize=(12,7))
plt.xlabel('Months')
plt.ylabel('passengers')
plt.show()


# In[20]:


# 3. Using Last Observation carried forward
df_air_last = pd.read_csv('airline-passenger-traffic.csv')
df_air_last.columns = ['Months','Passengers']
df_air_last['Months'] = df_air_last['Months'].astype('datetime64[ns]')
df_air_last.set_index('Months',inplace=True)


# In[21]:


# Here in method we use pad, ffill, bfill
df_air_last['Passengers'].fillna(method = 'pad',inplace=True)


# In[22]:


# Plot the graph
df_air_last.plot(figsize=(12,7))
plt.xlabel('Months')
plt.ylabel('passengers')
plt.show()


# ### Observation :-
# - Here we use LInear interpolation for missing value treatment bcz here in data trend is present.

# # Handling Outliers

# In[23]:


fig = plt.subplots(figsize=(12,2))
ax = sns.boxplot(df_air['Passengers'], whis=1.5)


# In[24]:


# Histogram plot

df_air['Passengers'].hist(figsize=(12,4))


# # Time Series Decomposition 

# ### Additive decomposition

# In[25]:


from statsmodels.api import tsa


# In[26]:


plt.figure(figsize=(12,8))
decomposiotion_add = tsa.seasonal_decompose(df_air.Passengers , model='additive') # additive seasonal index
decomposiotion_add.plot()
plt.show()


# ### Multiplicative Seasonal Decomposition

# In[27]:


decomposiotion_mul = tsa.seasonal_decompose(df_air.Passengers , model='multiplicative') # multiplicative seasonal index
decomposiotion_mul.plot()
plt.show()


# ### Observation :-
# - From Residual we can say that the series is Multiplicative.
# - Here in series Trens is present.

# - If only trend is present then you can go for relavent method like Simple moving average, Holt's method.
# - If trend and Season is present then you can go for relavent method like Holt's method, Holt's Winter or AR tech.

# # Built And Evaluate 

# ### Split data into training and test set 

# In[28]:


train_len = 120
train = df_air[0:train_len] # frist 120 months as training set
test = df_air[train_len:] # remaining 24 months as test set


# In[29]:


train


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


y_pred = train['Passengers'][train_len-1]
y_pred


# In[31]:


# adding new column to dataset:

train['y_pred']=y_pred


# In[32]:


train.head()


# # Simple time series method

# ### 1.Naive method
# - last observation carried froward

# In[33]:


df_naive = test.copy()


# In[34]:


df_naive['Naive_forecast'] = train['Passengers'][train_len-1]


# In[35]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'],label='Train')
plt.plot(test['Passengers'],label='Test')
plt.plot(df_naive['Naive_forecast'],label='Naive forecast')
plt.legend(loc='best')
plt.title('Naive Method')
plt.show()


# ### Calculate RMSE and MAPE

# In[36]:


from sklearn.metrics import mean_squared_error


# In[37]:


rmse = np.sqrt(mean_squared_error(test['Passengers'],df_naive['Naive_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-df_naive['Naive_forecast'])/test['Passengers'])*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# ### 2.Simple average method
# - Average of data carried froward

# In[38]:


df_avr = test.copy()


# In[39]:


df_avr['Avg_forecast'] = train['Passengers'].mean()


# In[40]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'],label='Train')
plt.plot(test['Passengers'],label='Test')
plt.plot(df_avr['Avg_forecast'],label='Average forecast')
plt.legend(loc='best')
plt.title('Average Method')
plt.show()


# ### Calculate RMSE and MAPE

# In[41]:


rmse = np.sqrt(mean_squared_error(test['Passengers'],df_avr['Avg_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-df_avr['Avg_forecast'])/test['Passengers'])*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# ### 3.Moving average method
# - Average of each window carried forward

# In[42]:


df_moving_avg = test.copy()
ma_window = 12


# In[43]:


# calculating average of moving window
df_moving_avg['Moving_avg_forecast'] = df_air['Passengers'].rolling(ma_window).mean()


# In[44]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'],label='Train')
plt.plot(test['Passengers'],label='Test')
plt.plot(df_moving_avg['Moving_avg_forecast'],label='Moving Average forecast')
plt.legend(loc='best')
plt.title('Moving Average Method')
plt.show()


# ### Calculate RMSE and MAPE

# In[45]:


rmse = np.sqrt(mean_squared_error(test['Passengers'],df_moving_avg['Moving_avg_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-df_moving_avg['Moving_avg_forecast'])/test['Passengers'])*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# # Exponential Smoothing Technique:

# ### 1. Simple Exponential Smoothing Technique:

# In[46]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model_simple_exp = SimpleExpSmoothing(train['Passengers'])
model_fit = model_simple_exp.fit(smoothing_level=0.2,optimized=False)
model_fit.params


# In[47]:


y_pred_simple_exp = test.copy()
y_pred_simple_exp['simple_exp_forecast'] = model_fit.forecast(24)


# In[48]:


plt.figure(figsize=(12,5))
plt.plot(train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_pred_simple_exp['simple_exp_forecast'], label='simple exponential soothing forecast')
plt.title('simple exponential smoothing method')
plt.legend(loc='best')
plt.show()


# #### Calculate RMSE and MAPE

# In[49]:


rmse = np.sqrt(mean_squared_error(test['Passengers'],y_pred_simple_exp['simple_exp_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_pred_simple_exp['simple_exp_forecast'])/test['Passengers'])*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# ### 2. Double [Holt's] Exponential Smoothing Technique:

# In[50]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[51]:


model_Holt_exp = ExponentialSmoothing(np.asarray(train['Passengers']), seasonal_periods=12, trend='multiplicative', seasonal=None)


# In[52]:


model_fit = model_Holt_exp.fit(smoothing_level=0.2, smoothing_slope=0.01, optimized=False)


# In[53]:


model_fit.params


# In[54]:


y_pred_Holt_exp = test.copy()
y_pred_Holt_exp['Double_exp_forecast']=model_fit.forecast(24)


# In[55]:


plt.figure(figsize=(12,5))
plt.plot(train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_pred_Holt_exp['Double_exp_forecast'], label='Holt\'s exponential smoothing forecast')
plt.title('Holt\'s exponential smoothing method')
plt.legend(loc='best')
plt.show()


# ### Calculating RMSE and MAPE

# In[56]:


rmse = np.sqrt(mean_squared_error(test['Passengers'],y_pred_Holt_exp['Double_exp_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_pred_Holt_exp['Double_exp_forecast'])/test['Passengers'])*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)


# ### 3. Holt's [triple] Winters Exponential Smoothing Technique:

# In[57]:


model_holt_winters = ExponentialSmoothing(np.asarray(train['Passengers']), seasonal_periods=12, trend='add', seasonal='mul')


# In[58]:


model_holt_winters = model_holt_winters.fit(optimized=True)


# In[59]:


model_holt_winters.params


# In[60]:


y_pred_Holt_winter = test.copy()
y_pred_Holt_winter['Holt_winter_forecast'] = model_holt_winters.forecast(24)


# ### Calculating RMSE and MAPE

# In[61]:


rmse = np.sqrt(mean_squared_error(test['Passengers'],y_pred_Holt_winter['Holt_winter_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_pred_Holt_winter['Holt_winter_forecast'])/test['Passengers'])*100,2)
print('RMSE :',rmse)
print('MAPE :',mape)

