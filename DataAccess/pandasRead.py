
from DataSets import pathReader as pr

import pandas as pd

def readForestFire():
    
    #Store package in DataSets\
    firePath = "/ForestFires/forestfires.csv"
    
    
    filePath = pr.returnDataSetPath(firePath)
    
    raw = pd.read_csv(filePath, header=0, engine='python')
    
    return raw