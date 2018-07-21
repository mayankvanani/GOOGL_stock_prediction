import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

## importing training set
df_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = df_train.iloc[:,1:2].values

## Feature scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
training_set_scaled = scale.fit_transform(training_set)

## Creating a data structure with 60 timestep an 1 output.
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0]) # first loop :(i=60) - Xtrain = 0 to 59 (as upper bound is excluded)
                                                   # second loop :(i=61) - Xtrain = 1 to 60
                                                   # [i-60:i, 0]
                                                   # rows   , columns
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

## Reshaping
# https://keras.io/layers/recurrent/   see the 'INPUT SHAPES' part so see the input arguments of 'newshape' 
X_train = np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))

### PART-2 Building LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1) ))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2)) 
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(rate=0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')

## Fitting the RNN to training set
regressor.fit(X_train, y_train, epochs=128, batch_size=32)

### PART-3 Making the prediction and visualising the results
# getting the real google stock price-2017 (test set)
df_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = df_test.iloc[:,1:2].values

df_total = pd.concat((df_train['Open'], df_test['Open']), axis=0)

inputs = df_total[len(df_total) - len(df_test) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = scale.transform(inputs)    
# creating array of 60 previous stock price for input for 20 test points in the test set.
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scale.inverse_transform(predicted_stock_price)

## Visualising the results
plt.plot(test_set, color='red', label='Real GOOGL stock price')
plt.plot(predicted_stock_price, color='blue', label='predicted GOOGL stock price')
plt.title('GOOGL STOCK PRICE PREDICTION')
plt.xlabel('Time')
plt.ylabel('google stock price')
plt.legend()
plt.show()










