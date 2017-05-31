
#Split a Data Frame into X train, Y target matrices
def splitDataFrame(dataFrame, target):
    	
    x = dataFrame.ix[:, dataFrame.columns != target]
    x = x.as_matrix(columns=x.columns)
    y = dataFrame.as_matrix(columns=[target])
    
    return x,y