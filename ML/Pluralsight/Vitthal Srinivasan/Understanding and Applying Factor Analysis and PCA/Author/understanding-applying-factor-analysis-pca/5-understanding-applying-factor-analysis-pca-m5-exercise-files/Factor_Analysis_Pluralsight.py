
import pandas as pd 
import numpy as np


combinedFile = 'C:/Users/admin/Downloads/Factor_Analysis/Returns.csv'



data=pd.read_csv(combinedFile ,sep=",")
data.head()

data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data = data.sort_values(['Date'], ascending=[True])

returns = data[[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64']]].pct_change()
returns = returns[1:]


yVars = returns[["AAPL","XOM","CVX","VLO","S&P","FVX"]]

stdYVars = (yVars - yVars .mean()) / (yVars .std())

stdReturns = (returns - returns .mean()) / (returns .std())

yVars .cov()
yVars .corr()

eig_val, eig_vec = np.linalg.eig(stdYVars.cov())

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])


eVector1 = eig_pairs[0][1]
eVector2 = eig_pairs[1][1]
eVector3 = eig_pairs[2][1]


pca1 = np.dot(stdYVars,eVector1.reshape(-1,1)).reshape(1,-1)
pca2 = np.dot(stdYVars,eVector2.reshape(-1,1)).reshape(1,-1)
pca3 = np.dot(stdYVars,eVector3.reshape(-1,1)).reshape(1,-1)


xData = np.array(zip(pca1.T,pca2.T,pca3.T)).reshape(-1,3)
yData = stdReturns["GOOG"]

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

goodGoogModel = linear_model.LinearRegression()
goodGoogModel .fit(xData,yData)
goodGoogModel .score(xData,yData)



model = sm.OLS(yData, xData)
results = model.fit()
print(results.summary())


