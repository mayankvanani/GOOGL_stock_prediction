# GOOGL_stock_prediction

## Summary
This dataset contains **GOOGL** stocks history of three years (2014-2015-2016). Here **Recurrent Neural Net-LSTM** is used to perform Time Series Analysis by looking at the past 60 observation to predict the future observation. The data set has columns of _Open, High, Low, Close, Volume_. The time series Analysis is performed just using **'Open'** Price. 

_(Source: Yahoo Finance)_

## Model-Graph

<img src="graph.jpg" width="440" alt="original">



## Further Improvements
This model has achieved an accuracy of **83%**. This can be further improved by dimension of the _Keras Tensor_ by considering all the columns and other MNCs/Competitors' Stock.
