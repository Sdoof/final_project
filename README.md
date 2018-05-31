# Final Project

### **Final project for Certificate in Python for Algorithmic Trading http://certificate.tpq.io/**

###### **Abstract**
Algorithmic trading involves computer programs coded with predefined trading instructions that automatically execute orders in response to changes in market data. Given the powerful packages now available the open source Python language allows for the efficient analysis of market data and incorportion of machine learning techniques. We will incoprorate these into testing a strategy which trades Australian dollar to US dollar (AUDUSD) Foreign Exchange rate.

Technical analysis has been used by market participants to determine patterns in price trends almost since markets have existed. Computers are highly efficient at determining patterns in data. We apply the Python echosystems machine learning capability to determine whether profitable price signals can be predicted using a number of widely used tehnical analysis indicators:

- past performance - lagged price data
- momentum - time series mometum
- MACD - momentum indicator showing the relationship between two moving averages
- RSI - momentum oscillator that measures the speed of price movements

We find that machine learning determines a model that produces profitable results before transaction costs and outperforms a simple buy and hold stategy. We also run the model in a a 'live' market environment.

##### **Jupyter Notebooks**

There are four Jupyter Notebooks and a python script included with the project

- 1) ONE_getdata_features.ipynb - retrieves data from Oanda API and creates the features to be used for testing
- 2) TWO_backtest_initial.ipynb - backtests the strategy on training and validation data
- 3) THREE_backtest_final.ipynb - once the strategy is finalised strategy is run on testing data which has not been previously seen
- 4) FOUR_tradingstrategy.ipynb - includes trading strategy class which runs strategy using live data from Oanda api. This is in testing mode and is set to stream data for 50 bars. The parameters are set so that trades will occur during this time for demonstration purposes (see **FOURTH NOTEBOOK** below)
- 5) FIVE_tradingstrategy_DEMO.py - a demo version of the python script of the trading strategy class this script is set to run for 10 minutes and trades will occur as the parameters have been set to shorter timeframe than those in the live model. 
- 6) FIVE_tradingstrategy_LIVE.py - a live python script of the trading strategy class this script is set to run as per the models parameters for 850 minutes. 

There are also a number of files that can be used with the project:

- 7) data.h5 - this is the raw historic data taken from Oanda
- 8) data_features - includes the raw historical data plus technical signals RSI, MACD and momentum used plus the lagged data which make up all of the features
- 9) final_model.sav - this is a pickle file of the model we have decided should be used for algorithmic trading following the backtest
- 10) jupyter_tradingstrategy.log - log file for Jupyter notebook **FOURTH NOTEBOOK** messages include information messages we have included in the class 
- 11) python_tradingstrategy.log - log file for Python script **FIVE PYTHON SCRIPT** messages include information messages we have included in the class 
- 12) liveresult_tradingstrategy.log - example log file from the model running for a 14 hour period on 20th May
- 13) tpqoa.py - self explanatory

##### **The backtest results**
It was decided to test a strategy on 10 minute historical data. After training the model on the backtest data and checking the results on the validation data as well as cross validating the parameters decided on were 20 bar lagged price returns, one lagged RSI and MACD bar as well as 10 bar and 50 bar momentum. When these parameters were run on the test data the results were promising.

There is a unforseen challenge with the decision to use 10 minute bars and the parameters chosen. When running the strategy on streaming data it takes approximately 7.5 hours before there is enough data for the model to begin trading given the parameters. This can be challenging in the debugging process. While next steps would be an investigation into how to append resampled streaming data to a historical data dataframe in the short term during testing and demonstration we would simply resample at 5s frequency rather than at 10 minute frequency.         

resample every 5 seconds for testing...
`self.dataresam = self.live_data.resample('5s', label = 'right').last().ffill().iloc[:-1]`
resample every 10 minutes as in model...
`self.dataresam = self.live_data.resample('10T', label = 'right').last().ffill().iloc[:-1]`

##### **The Final Project repository**

**FIRST NOTEBOOK**
The first notebook downloads the data we will use for research from the Oanda api. We determine to test the model on historical 10 minute bars for AUDUSD for the period 1 Jan 2018 - 30 April 2018. As we anticipate performing backtests numerous times we decide to save the data retrieved to an HDF5 file called 'data.h5' so that we dont repeatedly call the same data from the api. **Because of this the notebook can be run without needing the Oanda api data if after running the cells '1. Import' we skip straight to '4. Start with data saved from HDF5 file'.**

This notebook calculates the features that will be used in the analysis. These include Returns, Momentum, MACD and RSI Indices. The first notebook produces 30 lagged points for each of these items and the backtesting in the SECOND NOTEBOOK will select how many of them we will use in the model. Once the features have been produced they are saved for later use in the HDF5 file 'data_features.h5'.

**SECOND NOTEBOOK**
The second notebook performs the backtesting on the data. Firstly we load that data from our HDF5 file 'data_features.h5' Then we split the data in an attempt to help with the problem of overfitting. We firstly take 20% of the data away. **This data that has been removed is called the 'testing' set will not be used in the training and validation performed in this notebook but will instead be used in notebook three as a final test once the parameters of the model have been determined.**

Of the remaining 80% of the data we will split this in the proportion 80%/20% into a training set and a validation set. We will use the training set to train the model and then the validation set to determine what results are produced by the model on data it hasn't seen yet. We will use this process to come up with the best model we can by changing the features and the number of features (i.e. changed how many lagged data points we use).

The model we choose for this process is the Logistic Regression model from the Scikit Learn package. We also used a Support Vector Machine in the backtest as both models are good for clasification problems such as the one we are trying to solve (will the AUDUSD be up or down tomorrow?) We decided on logistic regression becasue it is good for problems where the classes are approximately linearly seperable, although similar to Logistic Regression SVM adds another dimension to achieve this making logistic regression the simpler of the two models for our purposes.

As well as having separated our data into training, validating and testing sets. We try some other techniques to help with the risk of overfitting our data. In particular we use Scikit Learns 'train-test_split' and 'five fold cross validation' techniques. The results of which can be seen in the notebook.

Finally having determined the optimal Logistic Regression model, we save the model using pickle, as the file 'final_model.sav'. We will use this saved model both in the testing step in notebook three and also use it in the streaming data in our algorithm in notebook four.

**THIRD NOTEBOOK**
In this third notebook we isolate the data (test set) we have in 'data_features.h5' that has been hidden from the model up until now. We use the model we have saved to pickle on the test set and see that the model performs resaonably well. It both makes a positive return and outperforms the buy and hold strategy in the test set. We are now ready to create our Python Algorithmic Trading Program.

**FOURTH NOTEBOOK**
The forth notebook is a python class that calls streaming data from the Oanda api. Having resampled the data into the appropriate 10 minute bars we used for the model, it puts the resampled data into the model and predicts trading signals based on the models parameters.

The tradingstategy() class has a number of methods:

- stream_data()       - streams data from the api ctx_stream and calls the method on_success()
- on_success()        - takes timestamp, bid and ask from streaming data into pandas dataframe, resmples the data into a new dataframe called 'dataresam'. We calculate returns and then call methods relative_strength(), macd(), prepare_features() and finally we call the method load_model().
- relative_strength() - calculates the RSI index
- macd()              - calculates the MACD Index
- prepare_features()  - creates a list of features columns 'cols' to be used with the 'dataresam' dataframe to pass the appropriate features to the model
- load_model()        - loads the model saved as a pickle file. Predicts trading signal. Calls execute_order()
- execute_order()     - based on prediction uses create_order() method from tpqoa helper class to enter market order
- close_out()         - if a prespecified number of streaming datapoints or amount of time has passed closes open orders and stops program

In constrast to the python script which runs for a certain amount of time (see below) the fourth notebook is set to run for a certain number of bars `rt.stream_data(stop = 50)`. For demonstration purposes the following settings are made self.mom1 = 2 rather than 10 determined in the model, self.mom2 = 10 rather than 50, resample set to 5s rather than 10T and `if self.ticks % 1 == 0:` rather than `% 50 == 0` so every tick is printed to the console.

It should be noted it is assumed the user will have to add the location for their own config file `rt = tradingstrategy('/root/pyalgo.cfg','AUD_USD')` in order to access the Oanda api.

##### **FIVE PYTHON SCRIPT**
Because of the issue highlighted over the choice of 10 minute bars there are two versions of the Python script included with the project.

**FIVE_tradingstrategy_DEMO** is configured for demonstration purposes rather than using the actual parameters need for the model. It resamples the streaming data every 5s, therefore creating 5 second bars, will run for 50 ticks `stop = 50`, print every tick to the console and uses 2 and 5 bar momentum.

**FIVE_tradingstrategy_LIVE** is configured using the actual parameters needed for the model. It will resample the streaming data every 10 minutes, therefore creating 10 minute bars, run for 850 minutes `stopmin = 850`, print every 50 ticks to the console and uses 10 and 50 bar momentum.

It should be noted it is assumed the user will have to add the location for their own config file `'/root/pyalgo.cfg'` in order to access the Oanda api.

`if __name__ == '__main__':
    di = tradingstrategy('/root/pyalgo.cfg','AUD_USD', stopmin = 850)`
    
**jupyter_tradingstrategy.log, python_tradingstrategy.log and liveresult_tradingstrategy.log**
We use the python logging library from its standard library. The script in both the fourth notebook and the Python script use logging for debugging and information purposes. For example at the end of the script a transaction summary is sent to the respective log file. An example of the logging results from the model running for a 14 hour period on 20th May can be found in the file liveresult_tradingstrategy.log.
                