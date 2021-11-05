import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from pmdarima import auto_arima                              # for determining ARIMA orders

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


'''

you'll often read in datasets and

they separate out the year and the month into columns like this.

So what we want to do is we want to figure out how we can use these two columns to create a date time

index.

'''
# Load dataset
df = pd.read_csv('../Data/co2_mm_mlo.csv')
df


'''
We'll create a new column called Date and say, PD to date time, and then we can actually pass in a

dictionary call of what the year should be and what the month should be.And then if we want what the day should be

'''
df["date"]=pd.to_datetime({"year":df["year"],"month":df["month"],"day":1})


'''
And now if we check out the head of our data frame, we notice we have this date and it looks like it's

now a time stamp object,
'''
df


'''
And you'll notice that the date column is, in fact, a date time object.
'''
df.info()


'''
Well, we still need to do, though, is we want this to actually be the index.

And now if I check the head of the data frame, I have my date index
'''

df.set_index("date")


'''
the last thing to do in order to use stats models is that my frequency.
'''

df.index.freq="MS"


'''
Let's go ahead and plot out this data.

You'll notice that the average column is sometimes missing a few values.

So what they did instead is, they just interpolated it between the previous points and some of the future

points to fill in that value.

So we'll go ahead and use the interpolated column that we were not missing any points.

you should definitely see here that there are some clear seasonality as well as some general upward trend.
'''
df["interpolated"].plot(figsize=(12,8))


'''
And to confirm that there are some seasonality, we can run a decomposition so we can say results,

go ahead and say seasonal decompose on that interpolated column.

And you can use either an additive model or a multiplicative model, but the key thing to note here

is we'll definitely see a clear seasonal component.

So when we plot out this result, we can see here the observed values, the general trend.

And definitely by the scale, it's going to be large enough that we want to take that into account,

which is why we're using a seasonal Arima model.
'''

period=int(len(df)/2)
result=seasonal_decompose(df["interpolated"],model='add',period=period);
result.plot();


'''
If you were unsure about your particular data set and the seasonality cycle (M) of when you should set what

you should basically set equal to.

You could take the seasonal component of this result and then expand that, plot it out into different

sizes and then judge from there.

Plot that out and then you could start looking and maybe zoom in on this to see at what point does the

seasonal cycle repeat itself.

So we can see here the certain repetition.

You would just zoom in and see how many rows that take.
'''

result.seasonal.plot(figsize=(12,8))


'''
So because of that, let's go ahead and run the auto arima in order to obtain the recommended orders.


we pass df['interpolated'] on to auto Arima and then we want to make sure that we specify seasonals equal to true, even

though that technically is the default and because we specified seasonals equal to true, we need to

make sure we state how many rows are there per period.

And in this case, the seasonal is happening every year.

So say M is equal to 12 since we have monthly data and there's 12 months per year.
'''
auto_arima(df['interpolated'],seasonal=True,m=12).summary()


'''
Let's go ahead and do a train test, split on the data and test that our model, see how it performs

in the test set and then forecast into the future.

So we have 729 rows.

Let's go ahead and set one year for testing.
'''
len(df)


'''
So that means our training is going to be the df.Loc from the beginning, all the way to 717.
'''

train=df.iloc[:717]
test=df.iloc[717:]


'''
So we're going to do now create the model.

We're going to pass in the interpolated column from the training data.

And here we're going to specify two parameters.

One is the first order for the Arima.

So AR, I, and MA of the normal Arima model.

That is going to be this first component here of (0,1,3).

And then the other one we're going to do is the seasonal order.

And that one's going to be the second one here, which is (1 0 1 12)

then we'll fit the model and get those results, so we'll a model

that fit.

Check the results summary.

And this is basically the same results or very similar results to what was just reported by Auto Arima,
'''

model=SARIMAX(train['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results = model.fit()
results.summary()


'''
Now, let's go ahead and get predicted values for our test set range.

And now let's create some predictions by simply calling results.predict( the start of the end and

we'll go ahead and say type is levels) to make sure we don't have any sort of issues of any difference

in components.

And then we'll rename it.
'''
start=len(train)
end = len(train)+len(test)-1
predictions=results.predict(start,end,type='levels').rename("SARIMA Predictions")
predictions





'''
So we run that, we have our predictions, so let's go ahead and plot them out against the test results,

here we can see the real blue interpolating results and the Sarino predictions.

So as you can tell for predicting about a year out, we're actually a pretty good.

Again, remember that our SARIMA model actually does not know what this data should be.

And we can see compared to the real data, it's pretty on target.
'''
# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''

ax = test['interpolated'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


'''
So if we actually want to evaluate the model, we can always do things such as import root, mean squared

error.

'''

from sklearn.metrics import  mean_squared_error
error=mean_squared_error(test['interpolated'],predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) MSE Error: {error:11.10}')



'''
And then my error would be something like our RMSE.

Test interpolated compared to our predictions,
'''
from statsmodels.tools.eval_measures import rmse
error=rmse(test['interpolated'],predictions)
print(f'SARIMA(0,1,3)(1,0,1,12) RMSE Error: {error:11.10}')


'''
Fit Entire data into model

And let's go on to predict one year into the future.

And we don't want to output the different results.

We want to put the true results in the same units as the original data.

so type='levels'

'''
model=SARIMAX(df['interpolated'],order=(0,1,3),seasonal_order=(1,0,1,12))
results=model.fit()
fcast = results.predict(len(df),len(df)+11,typ='levels').rename('SARIMA(0,1,3)(1,0,1,12) Forecast')



'''
So when we run that, we can see here at the very end what our forecast is.

'''

# Plot predictions against known values
title = 'Monthly Mean CO₂ Levels (ppm) over Mauna Loa, Hawaii'
ylabel='parts per million'
xlabel=''

ax = df['interpolated'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);



