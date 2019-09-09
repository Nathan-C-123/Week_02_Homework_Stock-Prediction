# import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics


## df = dataframe ##

# All data in csv
df_full = pd.read_csv("AAPL.csv")

# Set index to date (removes numbering)
df_full.set_index('Date', inplace=True)

# Output number of rows and columns
print(df_full.shape)

# Remove all columns other than date and Adj Close - set as new variable
df_raw = df_full[df_full.columns[-2]]

# Output first 5
#print(df_raw.head())

# Plot and show graph
df_raw.plot(label='AAPL', figsize=(16,8), title='Adj Close', grid=True, legend=True)
#df_full["Adj Close"].plot(label='AAPL', figsize=(16,8), title='Adj Close', grid=True, legend=True)
plt.show()


# Define the amount of days we're playing with
window_size=32
num_samples=len(df_full)-window_size

# Get indices of access for the data
indices=np.arange(num_samples).astype(np.int)[:,None]+np.arange(window_size+1).astype(np.int)

# Create the 2D matrix of training samples
data = df_full['Adj Close'].values[indices]

# Each row represents 32 days in the past
x = data[:,:-1]

# Each output value represents the 33rd day
y = data[:,-1]

# 80% used for training, 20% used for testing
split_fraction=0.8
ind_split=int(split_fraction*num_samples)

# Define arrays
x_train = x[:ind_split]
y_train = y[:ind_split]
x_test = x[ind_split:]
y_test = y[ind_split:]



# Error metrics
def get_errorVals (model_pred):
  #Function returns standard performance metrics
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, model_pred).round(4))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, model_pred).round(4))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, model_pred)).round(4))


# Scatter plots to visualize models
def get_plot (model_pred):
  plt.scatter(model_pred, y_test, color="gray")
  plt.plot(y_test, y_test, color='red', linewidth=2)
  plt.show()


# Model with Linear Regression, predict, get error values and plot
model_lreg = LinearRegression()
model_lreg.fit(x_train, y_train)
y_pred_lreg = model_lreg.predict(x_test)
print('Linear Regression:')
get_errorVals(y_pred_lreg)
get_plot(y_pred_lreg)
print('\n')


# Model with Ridge Regression, predict, get error values and plot
model_ridgereg = Ridge()
model_ridgereg.fit(x_train, y_train)
y_pred_ridgereg=model_ridgereg.predict(x_test)
print('Ridge Regression:')
get_errorVals(y_pred_ridgereg)
get_plot(y_pred_ridgereg)
print('\n')


# Model with Gradient Boosting Regression, predict, get error values and plot
model_gbreg = GradientBoostingRegressor()
model_gbreg.fit(x_train, y_train)
y_pred_gbreg = model_gbreg.predict(x_test)
print('Gradient Boosting Regression:')
get_errorVals(y_pred_gbreg)
get_plot(y_pred_gbreg)
print('\n')

# Convert to dataframe
df_compare=pd.DataFrame({
              "Linear regression":np.absolute(y_test-y_pred_lreg),
              "Ridge Regression":np.absolute(y_test-y_pred_ridgereg),
              "Gradient Boosting Regression":np.absolute(y_test-y_pred_gbreg)})

# Plot model errors by day and show
df_compare.plot.bar(title='Error by day', figsize=(16, 6))
# plt.ylim(0,10)
# plt.xlim(9,20)
plt.show()
