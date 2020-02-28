import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
import time
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#%matplotlib inline
#import datetime as dt
#from alpha_vantage.timeseries import TimeSeries
import yfinance as yf


class LSTMpredictor:
    """
    class to build Long Short Term Memory network for time-series in order to make prediction of future time-series
    This class includes
    - fetch stock prices via yahoo finance
    - split data into train and test data
    - build and fit LSTM
    - backtesting model via test set
    - making predictions of future trajectory
    """
    def __init__(self, interval = "daily", n_prev = 100, cut = 1000, verbose_plot = False):
        
        self.interval = interval #resolution of time-series
        self.n_prev = n_prev #size of input window sample
        self.cut = cut
        self.verbose_plot = verbose_plot

        #just optional, please use additional loaddata model!
    def loaddata(self, symbol, start_date = '1985-01-01'): 
        if self.interval == 'daily':
            xlabel = 'days'
            trj = yf.Ticker(symbol)
            trj = trj.history(period='max', start = start_date, interval = '1d')

        elif self.interval == 'weekly':
            xlabel = 'weeks'
            trj = yf.Ticker(symbol)
            trj = trj.history(period='max', start = start_date, interval = '1wk')

        elif self.interval == 'hourly':
            xlabel = 'hours'
            trj = yf.Ticker(symbol)
            trj = trj.history(interval = '1h')


        elif self.interval == 'minutely':
            xlabel = 'minutes'
            trj = yf.Ticker(symbol)
            trj = trj.history(period="7d", interval = '1m')
        trj = trj.reset_index()

        trj = trj.fillna(method='ffill')  

        if self.interval != 'minutely':

            trj = trj.replace(to_replace=0, method='ffill')

        if self.verbose_plot:
            plt.plot(trj.index, trj['Close'], label = symbol)
            plt.xlabel(xlabel, fontsize = 'x-large')
            plt.ylabel("close", fontsize = 'x-large')
            plt.title('Loaded Trajectory')
            plt.tick_params(labelsize="x-large")
            plt.legend(loc = 'best')
            #plt.savefig("run_figures/trj.png", bbox_inches='tight')
            plt.show()
            plt.close()
        
        return trj #returns pandas dataframe


    

    def create_ts(self, trj, value):
        ts = pd.DataFrame(trj[value],trj.index)

        # normalize features - 

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled = scaler.fit_transform(ts.values)
        ts = pd.DataFrame(scaled)
        series_s = ts.copy()
        for i in range(self.n_prev):
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
        self.cut = cut
        return train_X, train_y, test_X, test_y



    def build_fit_LSTM(self, trj, value, hidden_layers = 1, units = 32, dropout = 0.5, epochs = 100, batch_size = 32, filename = 'lstmpredicter.h5'):
        model = Sequential()
        model.add(LSTM(input_shape = (self.n_prev,1), output_dim= self.n_prev, return_sequences = True))
        model.add(Dropout(0.2))
        if hidden_layers >= 1: 
            for i in range(0,hidden_layers):#Add additional LSTM layer
                model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer="rmsprop")
        model.summary()

        model.compile(optimizer='rmsprop', loss="mse")

        timeseries, scaler = self.create_ts(trj = trj, value = value)

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
        model.save(filename)
        self.model = model
        self.scaler = scaler
        if self.verbose_plot:
            
            plt.plot(model.history.history["loss"], label = "loss")
            #plt.plot(model.history.history["val_loss"], label = "val_loss")
            plt.legend(loc = "best")
            plt.show()
            
        return train_X, train_y, scaler, test_X, test_y, model
    
    def backtest_Model(self, trj, value, length, filename = 'lstmpredicter.h5'):
        
        self.model = load_model(filename)
        
        #trj = self.loaddata(symbol = symbol, start_date = start_date)
        
        timeseries, scaler = self.create_ts(trj, value = value)
        
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

    def moving_test_window_preds(self, trj, value, n_steps, start, filename = 'lstmpredicter.h5'):
        
        self.model = load_model(filename)
        
        cut = start
        timeseries, scaler = self.create_ts(trj = trj, value = value)
        pred_trj=np.zeros(n_steps + cut) # Use this to store the prediction made on each test window
        pred_trj[:cut]=trj[value].values[:cut]
        for i in range(cut,n_steps+cut):
            X=(pred_trj[(i-self.n_prev):i]).reshape((1,self.n_prev))
            pred=self.model.predict(scaler.transform(X).reshape((1,self.n_prev,1)))
            pred_trj[i]=scaler.inverse_transform(pred)[0,0] # get the value from the numpy 2D array and append to predictions
        
        if self.verbose_plot:
            
            plt.plot(pred_trj[cut:], label = "prediction")  
            plt.plot(trj[value][cut:len(pred_trj)].values, label = "real")
            plt.ylabel(value)
            plt.xlabel("steps")
            plt.legend(loc = "best")
            plt.show()
            
        return pred_trj

    def pred_with_fit(self, trj, value, start, n_steps, steps_fit = 1, filename = 'lstmpredicter.h5'):
        
        self.model = load_model(filename)

        ''' n_steps - Represents the number of future predictions we want to make
                             This coincides with the number of windows that we will move forward
                             on the test data
        '''
        
        timeseries, scaler = self.create_ts(trj = trj, value = value)
        
        cut = start
        #cut = self.cut
        pred_trj=np.zeros(n_steps +cut) # Use this to store the prediction made on each test window
        pred_trj[:cut]=trj[value].values[:cut]
        
        if self.verbose_plot == False:
            verbose = 0
        else:
            verbose = 1
            
        for i in range(cut,n_steps+cut):
            print('step: ' + str(i-cut+1))
            X=(pred_trj[(i-self.n_prev):i]).reshape((1,self.n_prev))
            pred=self.model.predict(scaler.transform(X).reshape((1,self.n_prev,1)))
            pred_trj[i]=scaler.inverse_transform(pred)[0,0] # get the value from the numpy 2D array and append to predictions
            df = pd.DataFrame(pred_trj, columns = ['x'])
            if i % int(steps_fit) == 0:
                timeseries, scaler = self.create_ts(df[:i], value = "x")
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
