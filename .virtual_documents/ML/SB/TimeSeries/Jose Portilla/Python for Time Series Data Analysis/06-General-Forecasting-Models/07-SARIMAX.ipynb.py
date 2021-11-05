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


'''
And since we believe it to be weekly, given the plots that we just saw of, say,
M is equal to seven.
'''

# For SARIMA Orders we set seasonal=True and pass in an m value
auto_arima(df1['total'],seasonal=True,m=7).summary()


'''
we're first

going to fit to just a classic Suruma based model.

So we'll just take a seasonal or remodel and see how it performs.

So the first thing going to do is do a train test split.

So we're going to try to forecast a month into the future for restaurant visits, 

which means our test set should also be about a month.
'''
len(df1)


# Set four weeks for testing
train = df1.iloc[:436]
test = df1.iloc[436:]


'''

So we already split the data into a training set and a test set, and now we have our orders.

So it's time to actually fit this model.

now I'm only passing in the training data that it can fairly evaluate this model,


my order for the p,d,q terms, for the Arima portion of the model are just (1, 0, 0) which 

was reported back by auto Arima.

for the seasonal order of the model will go ahead say (1,0,1,7)

enforce_invertibility=False:

And then the last parameter I need to provide here is this parameter of 
inforce Inevitability.

Now, the reason we have to enforce convertibility equal to false here is mainly 

due to the way built stat's models library.

Essentially, we already know about the auto regression representation, where the most 
recent error can be written as a linear function of current and past observations.

So we already know that we can write out this linear function.

And the key part is for inconvertible process.Theta here is less than one.
And so the most recent observations have higher weight than observations from the more 
distant past.

Which makes sense, right?

That more recent data should hold the higher weight then further out data into the past.

However, when you have a process where theta is greater than one, then the weights 
increase as lags increase, which actually means the opposite.

That the more distance observations have greater influence on the current error, 
which is sometimes a peculiar situation.

And when Theta is equal to exactly one, then the weights are constant and size and the 
distance observations have the same influence as the recent observations.

So these last two situations typically don't make much sense, and so we prefer this 
inevitable process.

However, the way that stat's models has built out the Sorina X model internally 
It will try to force convertibility by forcing this theta to be less than one.
And in some particular situations, that actually doesn't make sense and it'll force an error.

So what we're going to do here is in order to avoid all those issues, we'll say 
in force convertibility equal to false and a way to understand whether or not you 
need to do that is simply run the model and see if you get the error.

And the error you get is called value error, non-convertible starting M.A parameters found.

So again, if you ever get the error.
When you're running one of these models that auto Auriemma suggested of value error, 
non-convertible starting M.A parameters found, that's totally OK.

Just inside your SARIMAX call, go ahead and say inforce inevitability equal to 
false and then that should remove that error.



'''

model=SARIMAX(train["total"],order=(1,0,0),seasonal_order=(1,0,1,7),enforce_invertibility=False)
results=model.fit()
results.summary()


'''
And we're going to do now is get predicted values into the future for our test set.

'''
start=len(train)
end=len(train)+len(test)-1

predictions=results.predict(start,end,dynamic=False).rename('SARIMA(1,0,0)(1,0,1,7) Predictions')
predictions


'''

So we can see here in that second week, we're really doing quite well.

But you know this for this particular week, there's a dip or maybe even a peak that 
we didn't actually grab.

So would it be interesting to see if for the situations where we didn't do such a 
great job predicting? 


And now we can see there was one holiday here that we didn't pick up on and there's 
three holidays here in the USA for this last one, and you'll notice it kind of like 
converges with these peaks.

So it'll be interesting to see if adding in the holidays as exogenous variables 
would actually improve our model.
'''

# Plot predictions against known values
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel=''

ax = test['total'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in test.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);


'''
The last thing I want to do is evaluate the model quantitatively 
using root mean squared error.
'''

from statsmodels.tools.eval_measures import mse,rmse

error1 = mse(test['total'], predictions)
error2 = rmse(test['total'], predictions)

print(f'SARIMA(1,0,0)(1,0,1,7) MSE Error: {error1:11.10}')
print(f'SARIMA(1,0,0)(1,0,1,7) RMSE Error: {error2:11.10}')


'''
So so far, we've only run screamo based models on our data with a seasonal 

component and the basic Arima components.

Now we're going to add in the exegonous variable.

In our case, it's going to be that holiday data.

'''

model = SARIMAX(train['total'],exog=train['holiday'],order=(1,0,0),seasonal_order=(1,0,1,7),enforce_invertibility=False)
results = model.fit()
results.summary()


# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
exog_forecast = test[['holiday']]  # requires two brackets to yield a shape of (35,1)
predictions = results.predict(start=start, end=end, exog=exog_forecast).rename('SARIMAX(1,0,0)(1,0,1,7)) Predictions')


# Plot predictions against known values
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel=''

ax = test['total'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in test.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);


# Print values from SARIMA above
print(f'SARIMA(1,0,0)(1,0,1,7) MSE Error: {error1:11.10}')
print(f'SARIMA(1,0,0)(1,0,1,7) RMSE Error: {error2:11.10}')
print()

error1x = mse(test['total'], predictions)
error2x = rmse(test['total'], predictions)

# Print new SARIMAX values
print(f'SARIMAX(1,0,0)(1,0,1,7)) MSE Error: {error1x:11.10}')
print(f'SARIMAX(1,0,0)(1,0,1,7)) RMSE Error: {error2x:11.10}')


model = SARIMAX(df1['total'],exog=df1['holiday'],order=(1,0,0),seasonal_order=(1,0,1,7),enforce_invertibility=False)
results = model.fit()
exog_forecast = df[478:][['holiday']]
fcast = results.predict(len(df1),len(df1)+38,exog=exog_forecast).rename('SARIMAX(1,0,0)(1,0,1,7)) Forecast')


# Plot the forecast alongside historical values
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel=''

ax = df1['total'].plot(legend=True,figsize=(16,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in df.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);



