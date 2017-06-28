from DataAccess import pandasRead
from sklearn.ensemble import RandomForestRegressor
import regressionEvaluator
import math

import pandas as pd

#Read raw Data Frame
rawDf = pandasRead.readForestFire()

#encode Data Frame, categorial data day,month
eDf = pd.get_dummies(rawDf, columns=['day','month'])

#setTarget
target = 'area'

#to much zeros transform 
eDf.area = eDf.area.apply(math.log1p)

#Use Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=500)

#evaluate model
regressionEvaluator.evaluate(eDf, rfr, target)
