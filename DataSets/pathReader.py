import os

def returnDataSetPath(dataSetPath):

    directory = os.path.dirname(os.path.abspath(__file__))
    
    return directory + dataSetPath

