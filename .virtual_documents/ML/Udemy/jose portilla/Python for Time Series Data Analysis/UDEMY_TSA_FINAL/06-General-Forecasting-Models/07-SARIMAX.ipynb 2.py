import pandas as pd
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")

# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")




# Load dataset
df = pd.read_csv('../Data/RestaurantVisitors.csv',index_col='date',parse_dates=True)
df.index.freq = 'D'


'''
OK, so for the exactions variables, we're going to see whether or not this holiday 
column actually affects this total column.

First off, we'll do a normal seasonal Arima model and then we'll add in this 
seasonal aroma exogenous model adding in this holiday information.

'''
df.head()


df.tail()


'''
So we're going to go ahead and drop that missing data since we can't really use 
it for training purposes.
'''
df1 = df.dropna()
df1.tail()


df1.columns


'''
So what we're going do here is we're going to change the data type of 
these last four columns, the restaurant numbers, as well as the total 
and make them into integers.



'''

# Change the dtype of selected columns
cols = ['rest1','rest2','rest3','rest4','total']
for col in cols:
    df1[col] = df1[col].astype(int)
df1.head()


'''
Recall that what we're trying to do is we're trying to predict and forecast the 

total number of visitors.So let's see what that looks like over time,

It does look like there's some sort of repeating peak.

It doesn't look like it's on a monthly level.

In fact, it's probably on a weekly level.

So maybe on the weekends there's a peak or maybe on Fridays or maybe on a happy hour day.
'''
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel='' # we don't really need a label here

ax = df1['total'].plot(figsize=(16,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


df1[df1['holiday']==1].index


'''
We don't really know for sure until we zoom in.

But what I'm also going to do is I'm going to write a little bit of code here so I can overlap when

there is a holiday and see, maybe there's an effect there, maybe not.

It may be a little hard to tell visually, but hopefully when we run, our Suruma X 

model will actually be able to tell that.


df1.query('holiday==1').index

So I'm essentially going to query the data frame for when Holliday's equal to one and 
grab those index numbers.

So if you return that, it basically returns back only the dates where the one happens 
to be equal to holiday.

ax.axvline(x=x, color='k', alpha = 0.3)

And what I want to do is for all these index locations at these date time stamps, 
I want to add in a little vertical line on my plot.


We'll say ax.axvline, which basically is a matplotlib command to add a vertical line onto
this axis object And then we'll say where X is equal today(x=x).

So essentially for every day in this date time index at a vertical line and then 
we can say what color we want it to be.

It's also the color equal to black, which is just color code, or you can also type out black.
And then let's give it a little bit of transparency.

'''

title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel='' # we don't really need a label here

ax = df1['total'].plot(figsize=(16,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in df1.query('holiday==1').index:# for days where holiday == 1 
                                        #or df1[df1['holiday']==1].index
       ax.axvline(x=x, color='k', alpha = 0.3)
       print(x)  ;# add a semi-transparent grey line
        


'''

So keep that in mind that just visually right now, it's a little unclear whether or not 

this exogenous variable is really going to be predictive of how many visitors 

show up to these restaurants.

However, intuitively, you should have some sort of idea that holidays probably do matter 

if there is a visit there or not.

Well, you also may want to do is since there is some indication of seasonality, 

is run a ETS decomposition.


So there is the observed values, 

the general trend, it looks like there's some sort of increase going on maybe during 

the holidays.A little hard to tell.

And there's definitely a seasonal component.
'''

result=seasonal_decompose(df1['total'])
result.plot();


'''
So very strong seasonal component.

In fact, let's just take a look at that seasonal component


we can see the peaks and valleys here, and what you

can do is eventually if you kind of zoom in on the seasonal component, you'll 

notice that it's it's weekly.

And in fact, you can kind of just tell that there's four seasonal periods per month 

indicating that there's four weeks per month.

So the seasonality of this data happens to be on a weekly basis, which makes sense.
'''
result.seasonal.plot(figsize=(15,5))


'''
we're first

going to fit to just a classic Suruma based model.

So we'll just take a seasonal or remodel and see how it performs.

So the first thing going to do is do a train test split.
'''
len(df1)


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


adf_test(df1['total'])


### Run <tt>pmdarima.auto_arima</tt> to obtain recommended orders
This may take awhile as there are a lot of combinations to evaluate.
