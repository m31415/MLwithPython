from DataAccess import pandasRead as pr
from DataPrep import standardizer
import pandas as pd

#Read raw Data Frame
rawDf = pr.readForestFire()

#encode Data Frame, categorial data day,month
eDf = pd.get_dummies(rawDf, columns=['day','month'])

#standardizer for NN or MR
sd = standardizer.sdizer(eDf.copy())

#normalized Data Frame
sd.normalize()
nDf = sd.dataFrame