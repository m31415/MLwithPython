import math

# R**2 Score 
def r2score(yPred,yTrue):
    
    u = ((yTrue- yPred)**2).sum()
    v = ((yTrue-yTrue.mean())**2).sum()
    r2score = (1 - (u/v))
    
    return r2score

#Mean absolute deviation MAD
def mad(yTrue, yPred):
    
    n = yTrue.size
    absoluteDelta = sum(abs(yTrue - yPred))
    mad = absoluteDelta * (1/n)
    
    return mad

# Root Mean Squared RMSE
def rmse(yTrue, yPred):
    
    n = yTrue.size
    squaredDelta = sum((yTrue - yPred)**2)
    rmse = math.sqrt(squaredDelta / n)
    
    return rmse
    