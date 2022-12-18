#!/usr/bin/env python
# coding: utf-8

# ### Simple Moving Average Indicator Using Machine Learning

# Here is used the TCS.NS stock market dataset of TCS. In this prediction build some signal 1(Yes) and No(0) to buy the stock or not. So whenever SMA10 is greater than SMA50 it will noted into the signal value as 1 and if not the signal value will be 0. After that, we build new column named ‘Position’, so whenever the pattern change the direction, it will store the data into the columns.

# In[1]:


# Import important liabraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


# In[2]:


# Read datasets for both training and testing: 
df=pd.read_csv("C:/Users/admin/Downloads/TCS.NS.csv")


# In[3]:


# Check the dataset shape
df.head()


# In[4]:


# Check datatypes for train data
df.info()


# In[5]:


# Check for missing data:
print(df.isna().sum())


# In[6]:


# Set the date columns as the index
df = df.set_index('Date')

# Set the index into datetime for efficency
df = df.set_index(pd.DatetimeIndex(df.index.values))


# In[8]:


df.info()


# In[9]:


df


# In[10]:


# Visualize the adj close price
plt.figure(figsize=(16,8))
plt.plot(df['Adj Close'], label='PGAS')

# Adding text into the visualization
plt.title('Adj Close Price History', fontsize=18)
plt.ylabel('Adj Close Price', fontsize=18)
plt.xlabel('Date', fontsize=18)

# Give legend
plt.legend()

# Show the graph
plt.show()


# In[11]:


# Create function for calculating Simple Moving Average (SMA)
def SMA(data, period=30, column='Adj Close'):
    return data[column].rolling(window=period).mean()


# In[12]:


# Create two new columns to store the 10 day and 50 day SMA
df['SMA10'] = SMA(df, 10)
df['SMA50'] = SMA(df, 50)


# In[13]:


print(df.SMA10[:15])


# In[14]:


# Get buy and sell signals
df['Signal'] = np.where(df['SMA10'] > df['SMA50'], 1, 0)
df['Position'] = df['Signal'].diff()
df['Buy'] = np.where(df['Position'] == 1, df['Adj Close'], np.NAN)
df['Sell'] = np.where(df['Position'] == -1, df['Adj Close'], np.NAN)


# In[15]:


# Visualize the close price with SMA also Buy and Sell Signals
plt.figure(figsize=(16,8))
plt.plot(df['Adj Close'], alpha=0.5, label='Actual Close Price')
plt.plot(df['SMA10'], alpha=0.5, label='SMA10')
plt.plot(df['SMA50'], alpha=0.5, label='SMA50')

# Make buy or sell signal
plt.scatter(df.index, df['Buy'], alpha=1, label='Buy Signal', marker='^', color='green')
plt.scatter(df.index, df['Sell'], alpha=1, label='Sell Signal', marker='v', color='red')

# Adding text into the visualization
plt.title('Adj Close Price w/ Buy and Sell Signals', fontsize=18)
plt.ylabel('Adj Close Price', fontsize=18)
plt.xlabel('Date', fontsize=18)

# Give legend
plt.legend()

# Show the graph
plt.show()


# In[16]:


df['Sell'].value_counts()


# In[17]:


df['Buy'].value_counts()


# In[18]:


df['Position'].value_counts()


# In[22]:


df['Signal'].value_counts()


# In[23]:


df['SMA10'].value_counts()


# In[24]:


df['SMA50'].value_counts()


# In[19]:


df.columns


# In[20]:


df.head(60)


# In[21]:


df.info()


# In[ ]:





# In[ ]:




