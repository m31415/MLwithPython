from DataAccess import pandasRead
from DataPrep import standardizer, dataFrameSplitter

from sklearn.ensemble import RandomForestRegressor

import pandas as pd

def score(yPred,yTrue):
    u = ((yTrue- yPred)**2).sum()
    v = ((yTrue-yTrue.mean())**2).sum()
    return (1 - (u/v))

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


print(score(yPred,y))