from DataAccess import pandasRead
from DataPrep import standardizer, dataFrameSplitter
from Metric import metrics
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

#Read raw Data Frame
rawDf = pandasRead.readForestFire()

#encode Data Frame, categorial data day,month
eDf = pd.get_dummies(rawDf, columns=['day','month'])

#standardizer for NN or MR
sd = standardizer.sdizer(eDf.copy())

#normalized Data Frame
sd.normalize()
nDf = sd.dataFrame

#Split Data Frame into X train, Y target as NumPy Arrays
x,y = dataFrameSplitter.splitDataFrame(nDf, 'area')

#Train randomForrest
rfr = RandomForestRegressor(n_estimators=500)
rfr.fit(x,y)

#predict 
yPred = rfr.predict(x)

#transpose y
y = y.sum(axis=1)


print(metrics.r2score(yPred,y))
print(metrics.mad(yPred,y))
print(metrics.rmse(yPred,y))