#
# tpqoa is a wrapper class for the
# Oanda v20 API (RESTful & streaming)
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import v20
import pandas as pd
import datetime as dt
import configparser

class tpqoa(object):
    ''' tpqoa is a Python wrapper class for the Oanda v20 API. '''

    def __init__(self, conf_file):
        ''' Init function expecting a configuration file with
        the following content:

        [oanda_v20]
        account_id = XYZ-ABC-...
        access_token = ZYXCAB...

        Parameters
        ==========
        conf_file: string
            path to and filename of the configuration file, e.g. '/home/me/oanda.cfg'
        '''
        self.config = configparser.ConfigParser()
        self.config.read(conf_file)
        self.access_token = self.config['oanda_v20']['access_token']
        self.account_id = self.config['oanda_v20']['account_id']
        self.ctx = v20.Context(
            hostname='api-fxpractice.oanda.com',
            port=443,
            ssl=True,
            application='sample_code',
            token=self.access_token,
            datetime_format='RFC3339')
        self.ctx_stream = v20.Context(
            hostname='stream-fxpractice.oanda.com',
            port=443,
            ssl=True,
            application='sample_code',
            token=self.access_token,
            datetime_format='RFC3339'
        )
        self.suffix = '.000000000Z'

    def get_instruments(self):
        ''' Retrieves and returns all instruments for the given account. '''
        resp = self.ctx.account.instruments(self.account_id)
        instruments = resp.get('instruments')
        instruments = [ins.dict() for ins in instruments]
        instruments = [(ins['displayName'], ins['name'])
                      for ins in instruments]
        return instruments

    def transform_datetime(self, dt):
        ''' Transforms Python datetime object to string. '''
        if isinstance(dt, str):
            dt = pd.Timestamp(dt).to_pydatetime()
        return dt.isoformat('T') + self.suffix

    def get_history(self, instrument, start, end,
                    granularity, price):
        ''' Retrieves historical data for instrument.

        Parameters
        ==========
        instrument: string
            valid instrument name
        start, end: datetime, str
            Python datetime or string objects for start and end
        granularity: string
            a string like 'S5', 'M1' or 'D'
        price: string
            one of 'A' (ask) or 'B' (bid)

        Returns
        =======
        data: pd.DataFrame
            pandas DataFrame object with data
        '''
        start = self.transform_datetime(start)
        end = self.transform_datetime(end)
        raw = self.ctx.instrument.candles(
            instrument=instrument,
            fromTime=start, toTime=end,
            granularity=granularity, price=price)
        raw = raw.get('candles')
        raw = [cs.dict() for cs in raw]
        for cs in raw:
            cs.update(cs['ask'])
            del cs['ask']
        if len(raw) == 0:
            return 'No data available.'
        data = pd.DataFrame(raw)
        data['time'] = pd.to_datetime(data['time'])
        data = data.set_index('time')
        data.index = pd.DatetimeIndex(data.index)
        for col in list('ohlc'):
            data[col] = data[col].astype(float)
        return data

    def create_order(self, instrument, units):
        ''' Places order with Oanda.

        Parameters
        ==========
        instrument: string
            valid instrument name
        units: int
            number of units of instrument to be bought (positive int, eg 'units=50')
            or to be sold (negative int, eg 'units=-100')
        '''
        request = self.ctx.order.market(
            self.account_id,
            instrument=instrument,
            units=units,
        )
        order = request.get('orderFillTransaction')
        print('\n\n', order.dict(), '\n')

    def stream_data(self, instrument, stop=None):
        ''' Starts a real-time data stream.

        Parameters
        ==========
        instrument: string
            valid instrument name
        '''
        self.stream_instrument = instrument
        self.ticks = 0
        response = self.ctx_stream.pricing.stream(
            self.account_id, snapshot=True,
            instruments=instrument)
        for msg_type, msg in response.parts():
            # print(msg_type, msg)
            if msg_type == 'pricing.Price':
                self.ticks +=1
                self.on_success(msg.time,
                                float(msg.bids[0].price),
                                float(msg.asks[0].price))
                if stop is not None:
                    if self.ticks >= stop:
                        break

    def on_success(self, time, bid, ask):
        ''' Method called when new data is retrieved. '''
        print(time, bid, ask)

    def get_account_summary(self, detailed=False):
        ''' Returns summary data for Oanda account.'''
        if detailed is True:
            response = self.ctx.account.get(self.account_id)
        else:
            response = self.ctx.account.summary(self.account_id)
        raw = response.get('account')
        return raw.dict()

    def get_transactions(self, tid=0):
        ''' Retrieves and returns transactions data. '''
        response = self.ctx.transaction.since(self.account_id, id=tid)
        transactions = response.get('transactions')
        transactions = [t.dict() for t in transactions]
        return transactions

    def print_transactions(self, tid=0):
        ''' Prints basic transactions data. '''
        transactions = self.get_transactions(tid)
        for trans in transactions:
             templ = '%5s | %s | %9s | %12s'
             print(templ % (trans['id'],
                            trans['time'],
                            trans['instrument'],
                            trans['units']))
