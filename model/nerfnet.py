# import stuff
import numpy as np
from copy import deepcopy

class nerfnet(object):
    def __init__( \
                self, \
                number_hidden_layers_=2, \
                factor_hidden_units_=2, \
                cost_type_='mse', \
                hidden_activation_type_='relu', \
                output_activation_type_=None, \
                solver_='sgd', \
                alpha_=1, \
                beta_=0, \
                threshold_=0.5, \
                batch_size_='auto', \
                max_iter_=2000, \
                tol_=0.0001, \
                early_stopping_=False, \
                epsilon_=1e-04 \
                ):

        # handle user input
        self.number_hidden_layers = number_hidden_layers_ # number of hidden layers
        self.factor_hidden_units = factor_hidden_units_ # factor of the #hidden units (compared to #input units)
        self.cost_type = cost_type_ # mse, ce, ...
        self.hidden_activation_type = hidden_activation_type_ # activation function of hidden layers
        self.output_activation_type = output_activation_type_ # activation function of output layer
        self.solver = solver_ # solver
        self.alpha = alpha_ # step width
        self.beta = beta_ # regularization factor
        self.threshold = threshold_ # threshold for classification applied on output of last layer
        self.batch_size = batch_size_ # batch size
        self.max_iter = max_iter_ # maximum number of iterations while training
        self.tol = tol_ # tolerance that limits the minimum improvement of cv cost before declaring eot
        self.early_stopping = early_stopping_ # use tolerance to stop training?
        self.epsilon = epsilon_ # distance to calculate numerical gradient

        # status flags on neural network
        self.weights_initialized = False

        # plant the seed
        np.random.seed(42)

        class MatrixStore:
            # open matrix store
            theta = {}
            a = {}
            z = {}
            delta = {}
            accu = {}

        # class sub classes
        self.matrices = MatrixStore()

    # init weights randomly. those will be trained / enable the nn to learn stuff.
    def init_weights(self, X_, y_):
        # determine weight size via given data
        inputUnits = X_.shape[1]
        hiddenUnits = inputUnits * self.factor_hidden_units
        # single output?
        if y_.ndim < 2:
            outputUnits = 1
        else:
            outputUnits = y_.shape[1]
        
        # choose optimal range
        amp = np.sqrt(6)/np.sqrt(hiddenUnits + inputUnits);

        # initialize weight matrices
        for t in range(0, self.number_hidden_layers + 1):
            key = str(t)
            if t == 0: # synapse between input and hidden layers (+1 for bias)
                self.matrices.theta[key] = 2*amp*np.random.random((hiddenUnits, inputUnits + 1)) - amp
            elif t < (self.number_hidden_layers): # synapse between hidden layers
                self.matrices.theta[key] = 2*amp*np.random.random((hiddenUnits, hiddenUnits + 1)) - amp            
            elif t == (self.number_hidden_layers): # synapse between hidden and output layer
                self.matrices.theta[key] = 2*amp*np.random.random((outputUnits, hiddenUnits + 1)) - amp
                
        # set weights flag ready!
        self.weights_initialized = True

                
    # propagate from input to output
    def propagate_forward(self, X_):
        # make sure input is 2d
        if X_.ndim < 2: # this is necessary if user wants to train or predict with single dataset
            X_ = X_[None,:]

        # define input as layer 0 (+ adding bias)
        temp_X = np.ones((X_.shape[0], X_.shape[1]+1))
        temp_X[:, 1:] = X_

        # assign input as activation of layer 0
        self.matrices.a['0'] = temp_X

        # loop through all hidden layers + output layer
        for t in range(1, self.number_hidden_layers + 2):
            self.matrices.z[str(t)] = np.dot(self.matrices.a[str(t-1)], self.matrices.theta[str(t-1)].T)
            # add bias if its a hidden layer
            if t < (self.number_hidden_layers + 1):
                temp_a = np.ones((self.matrices.z[str(t)].shape[0], self.matrices.z[str(t)].shape[1]+1))
                temp_a[:, 1:] = nerfnet.activation_function(self.hidden_activation_type, False, self.matrices.z[str(t)])
                self.matrices.a[str(t)] = temp_a
            else: # output layer
                self.matrices.a[str(t)] = nerfnet.activation_function(self.output_activation_type, False, self.matrices.z[str(t)])
     
    # calculate current error / cost.
    def calculate_cost(self, X_, y_):
        # get number of datasets
        m = X_.shape[0]

        # get hypothesis
        h = self.matrices.a[str(self.number_hidden_layers + 1)]

        if self.cost_type == 'ce':
            J = nerfnet.ce(m, y_, h)
        elif self.cost_type == 'mse':
            J = nerfnet.mse(m, y_, h)
            
            
        # add regularization to cost if regu para is not zero
        if self.beta != 0:
            collector = 0
            # loop over all thetas
            for theta in self.matrices.theta.values():
                # square and sum all values. leave bias out(!)
                collector += np.sum(np.square(theta[:, 1:]))

            # adjust cost
            J = J + (self.beta/(2*m)) * collector
            
        return J
    
    # propagate backwards from output to input
    def propagate_backwards(self, X_, y_):
        # get number of datasets
        m = X_.shape[0]

        # get hypothesis
        h = self.matrices.a[str(self.number_hidden_layers + 1)]

        # calculate delta of output layer
        self.matrices.delta[str(self.number_hidden_layers + 1)] = h - y_;

        # and for the rest of the layers (starting with the last)
        for t in range(self.number_hidden_layers, -1, -1):
            # input layer has no delta
            if t > 0:
                # get the derivative of activation fcn
                derivativeValue = nerfnet.activation_function(self.hidden_activation_type, True, self.matrices.z[str(t)])
                # calculate delta
                self.matrices.delta[str(t)] = np.dot(self.matrices.delta[str(t+1)], \
                                                self.matrices.theta[str(t)][:, 1:]) * \
                                                derivativeValue
            # fill the accumulators used to calculate the d/dTheta of J
            self.matrices.accu[str(t)] = np.dot(self.matrices.delta[str(t+1)].T, self.matrices.a[str(t)])/m
            # mind regularization if its used
            if self.beta != 0:
                self.matrices.accu[str(t)][:, 1:] += (self.beta / m) * self.matrices.theta[str(t)][:, 1:]

        # (d/dTheta) J
        dJ = np.array([])
        # stack all values ot accu dict to get gradient of J with respect to theta
        for layer in range(0, self.number_hidden_layers+1):
            dJ = np.hstack((dJ, np.reshape(self.matrices.accu[str(layer)], self.matrices.accu[str(layer)].size)))           
     
        return dJ
    
    # adjust the weights that will make the NN actually learn something.
    def adjust_weights(self):
        # use gradient descent.
        # loop through all layers
        for t in range(0, self.number_hidden_layers + 1):
            self.matrices.theta[str(t)] -= self.alpha * self.matrices.accu[str(t)]         
       
    # start training
    def training(self, X_train_, y_train_, X_cv_, y_cv_):
        # work the training data
        self.propagate_forward(X_train_)
        J_train = self.calculate_cost(X_train_, y_train_)
        # propagate back!
        self.propagate_backwards(X_train_, y_train_)
        # learn!
        self.adjust_weights()
        # work the cv data
        if X_cv_ is not None:
            self.propagate_forward(X_cv_)
            J_cv = self.calculate_cost(X_cv_, y_cv_)
        else:
            J_cv = -1
        
        return (J_train, J_cv)
                
    def trainingLoop(self, X_train_, y_train_, X_cv_, y_cv_):
        J_train = np.array([])
        J_cv = np.array([])
        currentTol = -1
        
        for epochs in range(0, self.max_iter):
            # create msg
            msg = 'Epoch: ' + str(epochs) + '/' + str(self.max_iter)
            
            # get J_cv only every nth epochs
            if (epochs%5) == 0:
                J = self.training(X_train_, y_train_, X_cv_, y_cv_)
                J_train = np.append(J_train, J[0])
                J_cv = np.append(J_cv, J[1])
                
                # print status
                msg += ' | Tol: ' + str(currentTol)
            else:
                self.training(X_train_, y_train_, None, None)

            # print out msg
            print(msg)
            
            # are there at least three J_cv's?
            if J_cv.size > 2:
                # calculate difference
                currentTol = (J_cv[-3] - np.amin(J_cv[-2:]))
                # is cv loss reaching convergence?
                if currentTol < self.tol:
                    # stop training if so!
                    break
                    
        return (J_train, J_cv)
        
    # function called by user to predict new output
    def predict(self, X_):
        # predict y given unknown X
        self.propagate_forward(X_)
        # return prediction
        return self.matrices.a[str(self.number_hidden_layers + 1)]     
    

    # function to check if back propagation works correctly
    def checkGradient(self, X_, y_):
        # call this function to compare gradient from backward
        # propagation and numerical gradient.
        # if there are just minor differences (~0.0001) backwards
        # propagation works fine.

        # propagate fwd
        self.propagate_forward(X_)

        # propagate backwards (receive gradient)
        grad = self.propagate_backwards(X_, y_)
        print(grad)
        # calculate numerical gradient
        numgrad = self.numericalGradient(X_, y_)
        print(numgrad)
        # calculate difference (compare gradients)
        diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)

        print(diff)
        
    # method to calculate the gradient numerically
    def numericalGradient(self, X_, y_):
        # init arrays to store values
        J = {}
        numericalGradient = np.array([])
        
        # loop through theta matrices
        for q in range(0, self.number_hidden_layers + 1):
            # loop through every single theta in weight matrix
            for p in np.nditer(self.matrices.theta[str(q)], op_flags=['readwrite']):
                # iterate + and -
                for t in range(0,2):
                    # make incremental change in one direction to single theta weight
                    p[...] += np.power(-1, t) * self.epsilon
                    # propagate fwd with modified weight matrix
                    self.propagate_forward(X_)
                    # calculate loss
                    J[t] = self.calculate_cost(X_, y_)
                    # restore original matrices
                    p[...] -= np.power(-1, t) * self.epsilon
                    
                tempGrad = (J[0]-J[1])/(2*self.epsilon)
                numericalGradient = np.append(numericalGradient, tempGrad)
        
        return numericalGradient
              
    def learningCurve(self, X_train_, y_train_, X_cv_, y_cv_, trainingEpochs_, stepSize_):
        # init arrays that will store costs
        J_train = np.array([])
        J_cv = np.array([])
        # init empty dict that will store original theta values
        origTheta = {}
        
        # store original weight matrices
        for p in range(0, self.number_hidden_layers + 1):
            origTheta[str(p)] = deepcopy(self.matrices.theta[str(p)])
                
        # loop over different training set sizes
        for m in range(0, len(X_train_), stepSize_):
            # get specific number of examples
            X_Data = X_train_[m:m+stepSize_]
            y_Data = y_train_[m:m+stepSize_]
            
            # train on that (a desired number of times)!
            for epochs in range(0, trainingEpochs_):
                self.training(X_Data, y_Data)
                
            # once training is done on that specific amount of examples, get the loss
            J_train = np.append(J_train, self.calculate_cost(X_Data, y_Data))
            # same thing on cv data (but all of it!)
            self.propagate_forward(X_cv_)
            J_cv = np.append(J_cv, self.calculate_cost(X_cv_, y_cv_))
            
            # unlearn network to start again with more examples!
            # store original weight matrices
            for p in range(0, self.number_hidden_layers + 1):
                self.matrices.theta[str(p)] = deepcopy(origTheta[str(p)])

            # calc and print progress
            progress = (m+stepSize_) * 100/len(X_train_)
            # cap progress (can overflow on last iteration!)
            if progress > 100:
                progress = 100
            print(('%.2f' % progress) + '%')
        
        return (J_train, J_cv)
            
    @staticmethod
    def activation_function(actvType_, derivativeFlag_, DataIn_):
        if actvType_ == 'sigmoid':
            if derivativeFlag_:
                DataOut = nerfnet.dsigmoid(DataIn_)
            else:
                DataOut = nerfnet.sigmoid(DataIn_)
        elif actvType_ == 'relu':
            if derivativeFlag_:
                DataOut = nerfnet.drelu(DataIn_)
            else:
                DataOut = nerfnet.relu(DataIn_)
        elif actvType_ == 'tanh':
            if derivativeFlag_:
                DataOut = nerfnet.dtanh(DataIn_)
            else:
                DataOut = nerfnet.tanh(DataIn_)
        else:
            DataOut = DataIn_

        return DataOut

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    @staticmethod
    def dsigmoid(z):
        return nerfnet.sigmoid(z) * (1 - nerfnet.sigmoid(z))
    
    @staticmethod
    def relu(z):
        return z * (z > 0)

    @staticmethod
    def drelu(z):
        return 1. * (z > 0)

    @staticmethod
    def tanh(z):
        return numpy.tanh(z)

    @staticmethod
    def dtanh(z):
        return 1. - z * z

    @staticmethod
    def ce(m_, y_, h_):
        # cost without regularization
        return (-1/m_) * np.sum(y_ * np.log(h_) + (1-y_) * np.log(1-h_));
        
    @staticmethod
    def mse(m_, y_, h_):
        # cost without regularization
        return (1/m_) * np.sum(np.square(y_ - h_))