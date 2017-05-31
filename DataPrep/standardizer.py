class sdizer:
    
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.means = []
        self.stds = []
     
    def normalize(self):
        
        for column in self.dataFrame.columns:
            
            mean = self.dataFrame[column].mean()
            self.means.append(mean)
            
            std = self.dataFrame[column].std()
            self.stds.append(std)
            
            self.dataFrame[column] = (self.dataFrame[column] - mean) / std
                     
    def denormalize(self):
        
        for column in self.dataFrame.columns:
            
            mean = self.means[0]
            self.means.pop(0)
            std = self.stds[0]
            self.stds.pop(0)
            self.dataFrame[column] = (self.dataFrame[column]  * std ) + mean