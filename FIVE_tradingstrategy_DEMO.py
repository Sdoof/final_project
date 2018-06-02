#
#
# script for running backtested model
# using logistic regression fit to historic
# data as predictor
#
# by Piers Watson
# as part of Certificate in Python for Algorithmic Trading
#
# THIS IS A DEMO MODEL ONLY: NOT BASED ON BACKTEST RESULTS
#
import sys
sys.path.insert(0,'/root/')
#
from tpqoa import tpqoa
import numpy as np
import pandas as pd
import tables as tb
import tstables as tstb
import pickle
import datetime
from datetime import timedelta
import time
from sklearn import linear_model

# create a logging file for debugging and information purposes
import logging    
logging.basicConfig(filename="python_tradingstrategy.log",level=logging.DEBUG,
                    format="%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s", 
                    datefmt ='%d/%m/%y %I:%M:%S %P')

class tradingstrategy(tpqoa):
    '''class for trading strategy using lagged returns, rsi and macd
    indicators. Optimised using logistic regression. Data streamed from
    Oanda api.
    '''
    # special method __init__ which is used to instantiate our python object
    def __init__(self, conf_file, instrument, stop):
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
        self.stop = stop
        self.stoptime = datetime.datetime(2018, 5, 30, 7, 30, 0, 0)
        self.model = linear_model.LogisticRegression()
        self.stream_data()
        
    def stream_data(self, stop = None):
        ''' starts a real time Oanda data stream
        '''
        print(20 * '-')
        print('trading starting at %12s' % self.starttime) 
        print('trading will stop after %12s ticks' % self.stop)
        print(20* '-')
        logging.info(20 * '-')
        logging.info('trading starting at %12s' % self.starttime) 
        logging.info('trading will stop after %12s ticks' % self.stop) 
        logging.info(20 * '-')
        self.ticks = 0
        response = self.ctx_stream.pricing.stream(self.account_id, snapshot = True,
                                                 instruments = self.instrument)
        for msg_type, msg in response.parts():
            if msg_type == 'pricing.Price':
                self.on_success(msg.time,
                              float(msg.bids[0].price),
                              float(msg.asks[0].price))
                if self.stop is not None:
                    if self.ticks >= self.stop:
                        self.close_out(self.stop)
                        break
                                            
    def on_success(self, time, bid, ask):
        '''Method called when new data is received. This updates the on_success
        method originally in the tpqoa class inherited by tradingstrategy class
        which merely printed and timestamped bid and ask prices'''
        self.ticks += 1
        if self.ticks % 1 == 0:
            print('%3d | %12s | bid %12s | ask %12s'% (self.ticks, time, bid, ask))
        # live streaming data append to live_data dataframe    
        self.live_data = self.live_data.append(pd.DataFrame({'bid':bid, 
                                                             'ask': ask},
                                                            index = [pd.Timestamp(time)]))
        # live streaming data resampled into bars
        # resample every 5 seconds for testing...
        self.dataresam = self.live_data.resample('5s', label = 'right').last().ffill().iloc[:-1]
        # resample every 10 minutes as in model...
        #self.dataresam = self.live_data.resample('10T', label = 'right').last().ffill().iloc[:-1]
        # having resmpled data calculate mid and therefore period return %
        self.dataresam['mid'] = self.dataresam.mean(axis=1)
        self.dataresam['returns'] = np.log(self.dataresam['mid'] / self.dataresam['mid'].shift(1))
        # if the length of the dataresam is large enough start calculating features
        if len(self.dataresam) > 22: 
            # call relative strength_method to calculate RSI Index
            self.dataresam['RSI'] = self.relative_strength(self.dataresam['mid'], self.rsi_n)
            # call macd method to calculate macd index
            self.dataresam['MACD'] = self.macd(self.dataresam['mid'])
            # call prepare_features method to calculate lagged, returns, RSI, macd and also mom
            self.dataresam = self.prepare_features(self.dataresam, self.lags)
            # call load_model method to use saved model
            self.load_model()
    
    def relative_strength(self, data, rsi_n):
        '''Creates RSI feature -
        initial RSI value created here
        '''
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
        '''creates macd feature'''
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
        '''model determined by backtesting has been saved in pickle
        we call the model here applying features created by live data
        '''
        LinMod = pickle.load(open('final_model.sav','rb'))
        pred = LinMod.predict(self.dataresam[self.cols])
        self.dataresam['prediction'] = pred
        # call method execute_order to submit trades to market
        self.execute_order()
        
    def execute_order(self):
        '''method to execute orders once predicted 
        by model
        '''
        # Entering long
        if self.dataresam['prediction'].iloc[-2] > 0 and self.position == 0:
            #print('going long')
            print('going long | %s | units %4d | ask %0.5f' % 
                         (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
            self.position = 1
            self.create_order(self.instrument, self.units)
            logging.info('going long | %s | units %4d | ask %0.5f' % 
                         (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
            
        elif self.dataresam['prediction'].iloc[-2] > 0 and self.position == -1:
            #print('covering short and going long')
            print('covering short and going long | %s | units %4d | ask %0.5f' %
                         (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
            self.position = 1
            self.create_order(self.instrument, 2 * self.units)
            logging.info('covering short and going long | %s | units %4d | ask %0.5f' %
                         (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
            
        # Entering short
        elif self.dataresam['prediction'].iloc[-2] < 0 and self.position == 0:
            #print('going short')
            print('going short | %s | units %4d | bid %0.5f' % 
                         (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))
            self.position = -1
            self.create_order(self.instrument, units = -self.units)
            logging.info('going short | %s | units %4d | bid %0.5f' % 
                         (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))
            
        elif self.dataresam['prediction'].iloc[-2] < 0 and self.position == 1:
            #print('covering long and going short')
            print('covering long and going short | %s | units %4d | bid %0.5f' %
                         (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))
            self.position = -1
            self.create_order(self.instrument, units = -2 * self.units)
            logging.info('covering long and going short | %s | units %4d | bid %0.5f' %
                         (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))      
                
    def close_out(self, stop):
        '''at end of test closes out open positions
        and prepares reporting of transactions. Note there is an id number 
        that needs to be updated so that previous data from Oanda not included
        in reporting see - UPDATE ID NUMBER HERE'''
        if self.ticks >= self.stop:

            logging.info('stop reached')
        
            # stop reached close out long position
            if self.position == 1:
                self.create_order(self.instrument, 
                                  units = -self.units)
                logging.info('stop reached - closing long, no open positions| %s | units %4d | bid %0.5f' 
                             % (self.instrument, self.units, self.dataresam.iloc[-1]['bid']))
                print(15 * '-')
                print('stop reached - closing long, no open positions') 
                print(15 * '-')
                self.position = 0
                
            # stop reached close out short position
            elif self.position == -1:
                self.create_order(self.instrument, units = self.units)
                logging.info('stop reached - closing short, no open position| %s | units %4d | ask %0.5f' 
                             % (self.instrument, self.units, self.dataresam.iloc[-1]['ask']))
                print(15 * '-')
                print('stop reached - closing short, no open positions')
                print(15 * '-')
                self.position = 0
            # id - needs to be set to the lastest order id in your Oanda transactions
            # report
            '''UPDATE ID NUMBER HERE'''
            self.response = self.ctx.transaction.since(self.account_id, id = 3064)
            self.transactions = self.response.get('transactions')
            for trans in self.transactions:
                trans = trans.dict()
                if trans['type']  == 'ORDER_FILL':
                    templ = '%s | id %8s | %6s | %9s | price %9s | p&l %8s'
                    print(templ % (trans['time'], trans['orderID'], trans['instrument'],trans['units'], 
                                   trans['price'], trans['pl']))
                    #               trans['fullPrice']['bids'][0]['price'],
                    logging.info(templ % (trans['time'], trans['orderID'], trans['instrument'],trans['units'],
                                          trans['price'], trans['pl'])) 
                    
if __name__ == '__main__':
    di = tradingstrategy('/root/pyalgo.cfg','AUD_USD', stop = 50)