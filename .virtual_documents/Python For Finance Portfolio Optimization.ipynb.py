#get_ipython().getoutput("pip install pandas-datareader")


from pandas_datareader import data as web
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#get the Stock symbol from portfolio
#FAANG

assets=['LTI.NS','BLUEDART.NS','TATAMOTORS.NS']

#assign weight to stocks
#all five stock to add upto=1 and assign equal amount of weights to the stocks
#20% of each of this stock in this portfolio 
weights=np.array([0.33,0.33,0.33])


#stock start and end date

stockStartDate='2013-01-01'
today=datetime.today().strftime('get_ipython().run_line_magic("Y-%m-%d')", "")
today


#create dataframe to store adjusted close price
df=pd.DataFrame()
#store adjusted close price of the stock into the datframe
for stock in assets:
    df[stock]=web.DataReader(stock,data_source='yahoo',start=stockStartDate,end=today)['Adj Close']


df


#visually show portfolio
my_stock=df
title='portfolio'
for c in my_stock.columns.values:
    plt.plot(my_stock[c],label=c)
plt.title=title
plt.xlabel('Date',fontsize=18)
plt.legend(my_stock.columns.values,loc='upper left')
plt.show()


#show daily simple return
returns=df.pct_change()
returns


#create and return annulize Covarience Matrix
#diagonal is varience , off diagonals are co-varience
#252 -> number of trading days in Year
cov_matrix_annual=returns.cov()*252
cov_matrix_annual


#portfolio varience
port_variance=np.dot(weights.T,np.dot(cov_matrix_annual,weights))
port_variance


#portfolio volatility aka standard devaition
port_volatility=np.sqrt(port_variance)
port_volatility


#calculate return of annual portfolio
#252 -> number of trading days in Year
portfolioSimpleAnnualReturn=np.sum(returns.mean() * weights) *252
portfolioSimpleAnnualReturn


#Show the expected annual return, volatility (risk), and variance


percent_var= str(round(port_variance, 2) * 100) +  'get_ipython().run_line_magic("'", "                                                                                                        ")

percent_vols = str(round(port_volatility, 2) * 100 )+ 'get_ipython().run_line_magic("'", "")

percent_ret = str(round (portfolioSimpleAnnualReturn, 2) * 100) + 'get_ipython().run_line_magic("'", "                                                                                                                                                                                                                                                                      ")

print ('Expected annual return: '+ percent_ret)

print ('Annual volatility /risk: '+ percent_vols)

print ('Annual variance: '+ percent_var)


#get_ipython().getoutput("pip install PyPortfolioOpt")


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import  expected_returns


#portfolio Optimization

#Calculate the Expected Return and annaualize sample covarience Matrix

mu=expected_returns.mean_historical_return(df)
mu


S=risk_models.sample_cov(df)
S


#optimize for Max Sharp Ratio
#Sharp ration : basically to describe how much excess return you receive for some amout of volatility
#it masure performance of investment compare to investment that is risk free

ef=EfficientFrontier(mu,S)
weights = ef.max_sharpe()
weights


cleaned_weights=ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


#get discrete allocation of each share per stock
from pypfopt import  DiscreteAllocation,get_latest_prices
latest_prices=get_latest_prices(df)
print(latest_prices)


weights=cleaned_weights
da=DiscreteAllocation(weights,latest_prices,total_portfolio_value=10000)

allocation,left_over=da.lp_portfolio()


print("Discrete Allocation : ", allocation)
print("Funds Remainning : ",left_over)
# Given the money 50000 to optimize it we can buy  {'RELIANCE.NS': 5, 'HDFCBANK.NS': 10, 'TCS.NS': 6} stock





#get_ipython().getoutput("export CVXOPT_BUILD_GLPK=1")
#get_ipython().getoutput("pip install cvxopt --no-binary cvxopt")



