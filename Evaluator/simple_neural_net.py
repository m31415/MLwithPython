# import stuff
import numpy as np

class simplenet(object):
    def __init__(self, X_, y_):
        # handle user input
        self.X_ = X_ # input training data
        self.y_ = y_ # output training data

        # plant the seed
        np.random.seed(42)

        class DefaultParams:
            # define params
            iterations = 10000
            alpha = 1
            regu = 0
            hiddenLayers = 2

        class MatrixStore:
            # open matrix store
            theta = {}
            a = {}
            z = {}
            delta = {}
            accu = {}

        # class sub classes
        self.params = DefaultParams()
        self.matrices = MatrixStore()
        
        # init weights
        self.init_weights()

    # init weights randomly. those will be trained / enable the nn to learn stuff.
    def init_weights(self):
        # determine weight size via given data
        inputUnits = self.X_.shape[1]
        hiddenUnits = inputUnits * 2
        outputUnits = self.y_.shape[1]
        
        # choose optimal range
        amp = np.sqrt(6)/np.sqrt(hiddenUnits + inputUnits);

        # initialize weight matrices
        for t in range(0, self.params.hiddenLayers + 1):
            key = str(t)
            if t == 0: # synapse between input and hidden layers (+1 for bias)
                self.matrices.theta[key] = 2*amp*np.random.random((hiddenUnits, inputUnits + 1)) - amp
            elif t < (self.params.hiddenLayers): # synapse between hidden layers
                self.matrices.theta[key] = 2*amp*np.random.random((hiddenUnits, hiddenUnits + 1)) - amp            
            elif t == (self.params.hiddenLayers): # synapse between hidden and output layer
                self.matrices.theta[key] = 2*amp*np.random.random((outputUnits, hiddenUnits + 1)) - amp
                
    # propagate from input to output
    def propagate_forward(self):
        # make sure input is 2d
        if self.X_.ndim < 2: # this is necessary if user wants to train or predict with single dataset
            self.X_ = self.X_[None,:]

        # define input as layer 0 (+ adding bias)
        temp_X = np.ones((self.X_.shape[0], self.X_.shape[1]+1))
        temp_X[:, 1:] = self.X_

        # assign input as activation of layer 0
        self.matrices.a['0'] = temp_X

        # loop through all hiddenlayers + output layer
        for t in range(1, self.params.hiddenLayers + 2):
            self.matrices.z[str(t)] = np.dot(self.matrices.a[str(t-1)], self.matrices.theta[str(t-1)].T)
            # add bias if its a hidden layer
            if t < (self.params.hiddenLayers + 1):
                temp_a = np.ones((self.matrices.z[str(t)].shape[0], self.matrices.z[str(t)].shape[1]+1))
                temp_a[:, 1:] = simplenet.sigmoid(self.matrices.z[str(t)])
                self.matrices.a[str(t)] = temp_a
            else:
                self.matrices.a[str(t)] = simplenet.sigmoid(self.matrices.z[str(t)])
     
    # calculate current error / cost.
    def calculate_cost(self):
        # get number of datasets
        m = self.X_.shape[0]

        # get hypothesis
        h = self.matrices.a[str(self.params.hiddenLayers + 1)]

        # cost without regularization
        J = (-1/m) * np.sum(self.y_ * np.log(h) + (1-self.y_) * np.log(1-h));

        # add regularization to cost if regu para is not zero
        if self.params.regu != 0:
            print("Regularization on!")
            collector = 0
            # loop over all thetas
            for theta in self.matrices.theta.values():
                # square and sum all values. leave bias out(!)
                collector += np.sum(np.square(theta[:, 1:]))

            # adjust cost
            J = J + (self.params.regu/(2*m)) * collector
        print(J)
    
    # propagate backwards from output to input
    def propagate_backwards(self):
        # get number of datasets
        m = self.X_.shape[0]

        # get hypothesis
        h = self.matrices.a[str(self.params.hiddenLayers + 1)]

        # calculate delta of output layer
        self.matrices.delta[str(self.params.hiddenLayers + 1)] = h - self.y_;

        # and for the rest of the layers (starting with the last)
        for t in range(self.params.hiddenLayers, -1, -1):
            # input layer has no delta
            if t > 0:
                self.matrices.delta[str(t)] = np.dot(self.matrices.delta[str(t+1)], \
                                                self.matrices.theta[str(t)][:, 1:]) * \
                                                simplenet.deri_sigmoid(self.matrices.z[str(t)])
            # fill the accumulators used to calculate the d/dTheta of J
            self.matrices.accu[str(t)] = np.dot(self.matrices.delta[str(t+1)].T, self.matrices.a[str(t)]) 
            # mind regularization if its used
            if self.params.regu != 0:
                self.matrices.accu[str(t)][:, 1:] += (self.params.regu / m) * self.matrices.theta[str(t)][:, 1:]

        # (d/dTheta) J
        dJ = np.array([])
        # stack all values ot accu dict to get gradient of J with respect to theta
        for accus in self.matrices.accu.values():
            dJ = np.hstack((dJ, accus.reshape(accus.size, order='F'))) if dJ.size else accus.reshape(accus.size, order='F')             
     
    # adjust the weights that will make the NN actually learn something.
    def adjust_weights(self):
        # use gradient descent.
        # loop through all layers
        for t in range(0, self.params.hiddenLayers + 1):
            self.matrices.theta[str(t)] -= self.params.alpha * self.matrices.accu[str(t)]         
       
    # start training
    def training(self):
        # iterate all the shit!
        for iter in range(self.params.iterations):
            self.propagate_forward()
            self.calculate_cost()
            self.propagate_backwards()
            self.adjust_weights()
                
    # function called by user to predict new output
    def predict(self, new_X_):
        # store input data from training
        stored_X = self.X_
        # set unknown input data
        self.X_ = new_X_
        # predict y given unknown X
        self.propagate_forward()
        # reset to old data
        self.X_ = stored_X
        # return prediction
        return self.matrices.a[str(self.params.hiddenLayers + 1)]     
    
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    @staticmethod
    def deri_sigmoid(z):
        return simplenet.sigmoid(z) * (1 - simplenet.sigmoid(z))
    
    
"""
# input example.
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# init simplenet
NN = simplenet(X,y)
# start training
NN.training()

# predict unknown value
NN.predict(np.array([0,0,0]))
"""