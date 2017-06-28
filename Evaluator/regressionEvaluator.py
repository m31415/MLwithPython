from DataPrep import standardizer, dataFrameSplitter
from Metric import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
#dF : pandas-DataFrame, encoded
#model : regression-model, fit(x, y), predict(x)
#target : target-columnName
def evaluate(dF, model, target):

    #standardizer for NN or MR
    sd = standardizer.sdizer(dF.copy())
    
    #normalized Data Frame
    sd.normalize()
    nDf = sd.dataFrame
    
    #Split Data Frame into X Data, Y Target as NumPy Arrays
    xData, yTarget = dataFrameSplitter.splitDataFrame(nDf, target)
    
    #Split between Test and Training Data
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yTarget, test_size=0.3, random_state=0)
    
    #Change to  a (x,)- Vector
    yTrain = yTrain.sum(axis=1)
    yTest = yTest.sum(axis=1)
    yTarget = yTarget.sum(axis=1)
    
    #Make own Scorer
    r2Scorer = make_scorer(metrics.r2score)
    
    #CV
    scores = cross_val_score(model, xTrain, yTrain, scoring=r2Scorer, cv=2)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    #Train model
    
    model.fit(xTrain, yTrain)
    
    #predict 
    yPred = model.predict(xTest)
    yPredTrain = model.predict(xTrain)
    
    #Score on TestData
    print(metrics.r2score(yPred,yTest))
    print(metrics.mad(yPred,yTest))
    print(metrics.rmse(yPred,yTest))
    
    #Score on TrainData
    print(metrics.r2score(yPredTrain,yTrain))
    print(metrics.mad(yPredTrain,yTrain))
    print(metrics.rmse(yPredTrain,yTrain))