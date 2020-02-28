import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%matplotlib inline
import datetime as dt
from alpha_vantage.timeseries import TimeSeries


class LSTMpredictor:
    """
    class to build Long Short Term Memory network for time-series in order to make prediction of future time-series
    This class includes
    - fetch stock prices via alpha_vantage
    - split data into train and test data
    - build and fit LSTM
    - backtesting model via test set
    - making predictions of future trajectory
    """
    def __init__(self, interval = "daily", n_prev = 100, cut = 1000, verbose_plot = False):
        
        self.interval = interval
        self.n_prev = n_prev
        self.cut = cut
        self.verbose_plot = verbose_plot

    def loaddata(self, symbol, key = 'X13823W0M7RN4DRR', interval = 'daily', verbose_plot = False):
        #check latest version of alpha_vantage!
    
        if interval == 'daily':
            xlabel = 'days'
            trj = TimeSeries(key=key, output_format='pandas')
            trj, trj_data = trj.get_daily(symbol, outputsize = "full")
            #trj['date'] = trj['index']
            #trj = trj.drop(['index'], axis=1)
            trj = trj.sort_values(by='date')
            trj = trj.reset_index()
            #trj = pd.DataFrame(trj['close'], trj.index, columns = ['x'])
        
        elif interval == 'hourly':
            xlabel = 'hours'
            trj = TimeSeries(key=key, output_format='pandas')
            trj, trj_data = trj.get_intraday(symbol, interval = "60min", outputsize = "full")
            #trj['date'] = trj['index']
            #trj = trj.drop(['index'], axis=1)
            trj = trj.sort_values(by='date')
            trj = trj.reset_index()
            #trj = pd.DataFrame(trj['close'], trj.index, columns = ['x'])

        elif interval == 'minutely':
            xlabel = 'minutes'
            trj = TimeSeries(key=key, output_format='pandas')
            trj, trj_data = trj.get_intraday(symbol, interval = "1min", outputsize = "full")
           # trj['date'] = trj['index']
           # trj = trj.drop(['index'], axis=1)
            trj = trj.sort_values(by='date')
            trj = trj.reset_index()
            #trj = trj.reset_index()

        elif interval == 'weekly':
            xlabel = 'weeks'
            trj = TimeSeries(key=key, output_format='pandas')
            trj, trj_data = trj.get_daily(symbol, outputsize = "full")
            #trj['date'] = trj['index']
            #trj = trj.drop(['index'], axis=1)
            trj = trj.sort_values(by='date')
            trj = trj.reset_index()
            trj['days'] = trj.index
            trj=trj.assign(weeks=np.floor(trj["days"]/7).astype(int))
            ts_o=trj.copy()
            ts_o=ts_o.assign(date=ts_o.index)
            trj=trj.groupby("weeks").mean() #important to make sure consistent time step
            trj=trj.assign(date=ts_o.groupby("weeks")["date"].apply(lambda x: np.min(x)))

        if verbose_plot:
            plt.plot(trj.index, trj['4. close'], label = symbol)
            plt.xlabel(xlabel, fontsize = 'x-large')
            plt.ylabel("close", fontsize = 'x-large')
            plt.title('Loaded Trajectory')
            plt.tick_params(labelsize="x-large")
            plt.legend(loc = 'best')
            #plt.savefig("run_figures/trj.png", bbox_inches='tight')
            plt.show()
            plt.close()

        return trj #returns a pandas data frame

    

    def create_ts(self, trj, value, n_prev):
        ts = pd.DataFrame(trj[value],trj.index)

        # normalize features - 

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(ts.values)
        ts = pd.DataFrame(scaled)
        series_s = ts.copy()
        for i in range(n_prev):
            ts = pd.concat([ts, series_s.shift(-(i+1))], axis = 1)

        ts.dropna(axis=0, inplace=True)
        return ts, scaler


    def train_test(self, ts, cut): 
        train = ts.iloc[:cut, :]
        test = ts.iloc[cut:,:]


        train = shuffle(train)

        train_X = train.iloc[:,:-1]
        train_y = train.iloc[:,-1]
        test_X = test.iloc[:,:-1]
        test_y = test.iloc[:,-1]

        train_X = train_X.values
        train_y = train_y.values
        test_X = test_X.values
        test_y = test_y.values

        train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
        test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)

        return train_X, train_y, test_X, test_y



    def build_fit_LSTM(self, trj, value, units = 32, dropout = 0.5, epochs = 100, batch_size = 32):
        model = Sequential()
        model.add(LSTM(input_shape = (self.n_prev,1), output_dim= self.n_prev, return_sequences = True))
        model.add(Dropout(0.2))
        if units >= 1: #Add additional LSTM layer
            model.add(LSTM(units))
            model.add(Dropout(dropout))
        model.add(Dense(1))
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer="rmsprop")
        model.summary()

        model.compile(optimizer='rmsprop', loss="mse")

        timeseries, scaler = self.create_ts(trj = trj, value = value, n_prev = self.n_prev)

        train_X, train_y, test_X, test_y = self.train_test(timeseries, cut = self.cut)

        initial_weights=model.get_weights()
        

        start = time.time()
        if self.verbose_plot == False:
            verbose = 0
        else:
            verbose = 1
        model.fit(train_X, train_y, epochs=epochs, shuffle=True, batch_size = batch_size, verbose = verbose)
        #model.fit(train_X, train_y, epochs=100, shuffle=True, 
              #validation_data=(test_X, test_y), batch_size = 32)
        print("> Compilation Time : ", time.time() - start)
        model.save("lstmpredicter.h5")
        self.model = model
        self.scaler = scaler
        if self.verbose_plot:
            
            plt.plot(model.history.history["loss"], label = "loss")
            #plt.plot(model.history.history["val_loss"], label = "val_loss")
            plt.legend(loc = "best")
            plt.show()
            
        return train_X, train_y, scaler, test_X, test_y, model
    
    def backtest_Model(self, symbol, value, length):
        trj = self.loaddata(symbol = symbol, key = 'X13823W0M7RN4DRR', interval = self.interval, verbose_plot = self.verbose_plot)
        timeseries, scaler = self.create_ts(trj, value = value, n_prev = self.n_prev)
        
        if length<= len(trj):
            train_X, train_y, test_X, test_y = self.train_test(timeseries[:length], cut = 0)
        else:
            print("please choose length smaller than " + str(len(trj)) + " !")
            
        preds = self.model.predict(test_X)
        preds = scaler.inverse_transform(preds)
        
        actuals = trj[value][self.n_prev:length+self.n_prev].values
        
        plt.plot(actuals, label = "real")
        plt.plot(preds, label = "prediction")
        plt.ylabel(value)
        plt.xlabel(self.interval)
        plt.legend(loc = "best")
        plt.show()
        
        return trj, actuals, preds

    def moving_test_window_preds(self, trj, value, n_steps):

        ''' n_steps - Represents the number of future predictions we want to make
                             This coincides with the number of windows that we will move forward
                             on the test data
        '''
        
        timeseries, scaler = self.create_ts(trj = trj, value = value, n_prev = self.n_prev)
        pred_trj=np.zeros(n_steps + self.cut) # Use this to store the prediction made on each test window
        pred_trj[:self.cut]=trj[value].values[:self.cut]
        for i in range(self.cut,n_steps+self.cut):
            X=(pred_trj[(i-self.n_prev):i]).reshape((1,self.n_prev))
            pred=self.model.predict(scaler.transform(X).reshape((1,self.n_prev,1)))
            pred_trj[i]=scaler.inverse_transform(pred)[0,0] # get the value from the numpy 2D array and append to predictions
        
        if self.verbose_plot:
            
            plt.plot(pred_trj[self.cut:], label = "prediction")  
            plt.plot(trj[value][self.cut:len(pred_trj)].values, label = "real")
            plt.ylabel(value)
            plt.xlabel("steps")
            plt.legend(loc = "best")
            plt.show()
        return pred_trj

    def pred_with_fit(self, trj, value, n_steps, steps_fit = 1):

        ''' n_steps - Represents the number of future predictions we want to make
                             This coincides with the number of windows that we will move forward
                             on the test data
        '''
        timeseries, scaler = self.create_ts(trj = trj, value = value, n_prev = self.n_prev)
        
        cut = self.cut
        pred_trj=np.zeros(n_steps +cut) # Use this to store the prediction made on each test window
        pred_trj[:cut]=trj[value].values[:cut]
        
        if self.verbose_plot == False:
            verbose = 0
        else:
            verbose = 1
            
        for i in range(cut,n_steps+cut):
            print('step: ' + str(i-cut))
            X=(pred_trj[(i-self.n_prev):i]).reshape((1,self.n_prev))
            pred=self.model.predict(scaler.transform(X).reshape((1,self.n_prev,1)))
            pred_trj[i]=scaler.inverse_transform(pred)[0,0] # get the value from the numpy 2D array and append to predictions
            df = pd.DataFrame(pred_trj, columns = ['x'])
            if i % int(steps_fit) == 0:
                timeseries, scaler = self.create_ts(df[:i], value = "x", n_prev = self.n_prev)
                train_X, train_y, test_X, test_y = self.train_test(timeseries, cut = len(pred_trj))
                self.model.fit(train_X, train_y, epochs=5, shuffle=True, batch_size = 512, verbose = verbose)
        if self.verbose_plot:
            plt.plot(pred_trj[cut:], label = "prediction")  
            plt.plot(trj[value][cut:len(pred_trj)].values, label = "real")
            plt.ylabel(value)
            plt.xlabel("steps")
            plt.legend(loc = "best")
            plt.show()
            
        return pred_trj
    
    def to_years(self, week, t0):
        return t0.year+(t0.isocalendar()[1]+week)/52.1429

    def week_range(self, n_weeks, t0):
        return to_years(np.arange(n_weeks), t0)
