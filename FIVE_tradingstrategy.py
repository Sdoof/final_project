from tpqoa import tpqoa
import numpy as np
import pandas as pd
import tables as tb
import tstables as tstb
import pickle
import datetime
import time
import sys
sys.path.insert(0, '/root/')
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.style.use('seaborn')
%matplotlib inline

import logging    
logging.basicConfig(filename="jupyter_tradingstrategy.log",level=logging.DEBUG,
                    format="%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s", 
                    datefmt ='%d/%m/%y %I:%M:%S %P')

class tradingstrategy(tpqoa):
    '''class for trading strategy using lagged returns, rsi and macd
    indicators. Optimised using logistic regression.
    '''
    def __init__(self, conf_file, instrument):
        tpqoa.__init__(self, conf_file)
        self.instrument = instrument
        self.live_data = pd.DataFrame()
        self.position = 0
        self.ticks = 0
        self.units = 100000
        self.rsi_n = 21
        self.mom1 = 2
        self.mom2 = 5
        self.lags = 20
        self.model = linear_model.LogisticRegression()
        
    def stream_data(self, stop = None):
        # self.stoptime = datetime.datetime(2018, 5, 25, 4, 0, 0, 0)
        ''' starts a real time Oanda data stream'''
        self.ticks = 0
        response = self.ctx_stream.pricing.stream(self.account_id, snapshot = True,
                                                 instruments = self.instrument)
        for msg_type, msg in response.parts():
            if msg_type == 'pricing.Price':
                self.on_success(msg.time,
                              float(msg.bids[0].price),
                              float(msg.asks[0].price))
                if stop is not None:
                    if self.ticks >= stop:
                        self.close_out(stop)
                        break
                            
    def on_success(self, time, bid, ask):
        '''Method called when new data is received. This updates the on_success
        method originally in the tpqoa class inherited by tradingstrategy class
        which merely printed and timestamped bid and ask prices'''
        self.ticks += 1
        if self.ticks % 1 == 0:
            print('%3d | '% self.ticks, time, bid, ask)
            
        self.live_data = self.live_data.append(pd.DataFrame({'bid':bid, 
                                                             'ask': ask},
                                                            index = [pd.Timestamp(time)]))
        self.dataresam = self.live_data.resample('5s', label = 'right').last().ffill().iloc[:-1]
        self.dataresam['mid'] = self.dataresam.mean(axis=1)
        self.dataresam['returns'] = np.log(self.dataresam['mid'] / self.dataresam['mid'].shift(1))
        if len(self.dataresam) > 22: # self.mom2: ******************************
                                                # *****************************
            self.dataresam['RSI'] = self.relative_strength(self.dataresam['mid'], self.rsi_n)
            self.dataresam['MACD'] = self.macd(self.dataresam['mid'])
            self.dataresam = self.prepare_features(self.dataresam, self.lags)
            self.load_model()
    
    def relative_strength(self, data, rsi_n):
        '''Creates RSI feature -
        initial RSI value created here'''
        abchange = (data - data.shift(1)) # calculate absolute daily change
        rsperiod = abchange[:rsi_n + 1]
        upday = rsperiod[rsperiod >= 0].sum() / rsi_n # in the RSI period what is the up day change
        dnday = -rsperiod[rsperiod < 0].sum() / rsi_n # in the RSI period what is the down day change
        rs = upday / dnday # up day change/down day change ratio
        rsi = np.zeros_like(data)
        rsi[:rsi_n] = 100. - (100. / ( 1. + rs)) # formula for RSI Index calculation
        
        '''calculates subsequent change in RSI values'''
        for i in range(rsi_n, len(data)):
            abchg = abchange[i - 1]
            if abchg > 0:
                upval = abchg
                dnval = 0
            else:
                upval = 0
                dnval = abs(abchg)
            
            # iterate through each daily change proportionally adding it
            # to the respective RSI period change
            upday = (upday * (rsi_n - 1) + upval) / rsi_n
            dnday = (dnday * (rsi_n - 1) + dnval) / rsi_n
            
            rs = upday / dnday # up day change/down day change ratio
            rsi[i] = 100. - (100. / ( 1. + rs)) # formula for RSI Index calculation
        rsi = pd.DataFrame(rsi)
        rsi.index = data.index
        rsi.columns = ['RSI']
        return rsi # Return the RSI Index value calculated
    
    def macd(self, data, slow = 26, fast = 12, signal = 9):
        # calculate respective fast and slow exponential moving averages
        ema_fast = data.ewm(span = fast).mean()
        ema_slow = data.ewm(span = slow).mean()
        # MACD line is slow m.a. minus fast m.a.
        macd_line = ema_slow - ema_fast
        # signal line is 9 day ema of macd line
        sig_line = macd_line.ewm(span = signal).mean()
        # macd histogram is the macd line minus the signal line
        macd_hist = macd_line - sig_line
        macd_hist = pd.DataFrame(macd_hist)
        macd_hist.columns = ['MACD']
        return macd_hist
            
    def prepare_features(self, df, lagz):
        '''creates lagged and momentum features'''
        self.cols = []
        
        #self.features = ['RSI','MACD','Returns']
        # add lagged RSI and MACD data, backtest suggests 1
        # lagged return only
        for feat1 in ['RSI','MACD']:
            lag1 = 1
            col = '%s_lag_%d' % (feat1, lag1)
            df[col] = df[feat1].shift(lag1)
            self.cols.append(col)
            
        # add lagged return data, backtest suggests 20
        # lagged returns        
        for lag in range(1,lagz + 1):
            col = 'Returns_lag_%d' % lag
            df[col] = df['returns'].shift(lag)
            self.cols.append(col)
            
        # add short term momentum signal
        df['MOM1'] = np.where(df['returns'].rolling(self.mom1).mean() > 0, 1, 0)
        df['MOM1'] = df['MOM1'].shift(1)
        self.cols.append('MOM1')
        # add long term momentum signal
        df['MOM2'] = np.where(df['returns'].rolling(self.mom2).mean() > 0, 1, 0)
        df['MOM2'] = df['MOM2'].shift(1)
        self.cols.append('MOM2')
        df.dropna(inplace = True)
        return df
    
    def load_model(self):
        LinMod = pickle.load(open('final_model.sav','rb'))
        pred = LinMod.predict(self.dataresam[self.cols])
        self.dataresam['prediction'] = pred
        self.execute_order()
        
    def execute_order(self):
        # Entering long
        if self.dataresam['prediction'].iloc[-2] > 0 and self.position == 0:
            print('going long')
            self.position = 1
            self.create_order(self.instrument, self.units)
            logging.info('going long | %s | units %4d | ask %0.5f' % 
                         (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
            
        elif self.dataresam['prediction'].iloc[-2] > 0 and self.position == -1:
            print('covering short and going long')
            self.position = 1
            self.create_order(self.instrument, 2 * self.units)
            logging.info('covering short and going long | %s | units %4d | ask %0.5f' %
                         (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
            
        # Entering short
        elif self.dataresam['prediction'].iloc[-2] < 0 and self.position == 0:
            print('going short')
            self.position = -1
            self.create_order(self.instrument, units = -self.units)
            logging.info('going short | %s | units %4d | bid %0.5f' % 
                         (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))
            
        elif self.dataresam['prediction'].iloc[-2] < 0 and self.position == 1:
            print('covering long and going short')
            self.position = -1
            self.create_order(self.instrument, units = -2 * self.units)
            logging.info('covering long and going short | %s | units %4d | bid %0.5f' %
                         (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))      
                
    def close_out(self, stop):
        if self.ticks >= stop:
            logging.info('stop reached')
        
            # stop reached close out long position
            if self.position == 1:
                self.create_order(self.instrument, 
                                  units = -self.units)
                logging.info('stop reached - closing long, no open positions| %s | units %4d | bid %0.5f' 
                             % (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))
                print('stop reached - closing long, no open positions') 
                self.position = 0
                
            # stop reached close out short position
            elif self.position == -1:
                self.create_order(self.instrument, units = self.units)
                logging.info('stop reached - closing short, no open position| %s | units %4d | ask %0.5f' 
                             % (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
                print('stop reached - closing short, no open positions')
                self.position = 0
        
            print(15 * '-')
            # sys.exit('Trading has stopped')