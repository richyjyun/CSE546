'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.degree = degree
        self.reg_lambda = reg_lambda
        self.theta = None
        self.avg = None     # for normalization
        self.std = None     # for normalization

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """

        # Polynomial expansion
        n = len(X)
        xpoly = np.empty([n, degree])
        for x in range(0, n):
            for d in range(0, degree):
                xpoly[x, d] = np.power(X[x], d+1)

        return xpoly

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        # Polynomial expansion
        x = self.polyfeatures(X, self.degree)

        # Normalization
        self.avg = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        x = x-self.avg[None, :]
        x = x/self.std[None, :]

        # From linreg_closedform.py
        ###############################
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), x]

        n, d = X_.shape
        d = d - 1  # remove 1 for the extra column of ones we added to get the original num features

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)
        ###############################

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """

        # Polynomial expansion
        x = self.polyfeatures(X, self.degree)

        # Normalization
        x = x - self.avg[None, :]
        x = x / self.std[None, :]

        # From linreg_closedform.py
        ###############################
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), x]

        # predict
        return X_.dot(self.theta)
        ###############################

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTest[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape

    reg = PolynomialRegression(degree, reg_lambda)

    for i in range(1, n):
        reg.fit(Xtrain[0:i+1], Ytrain[0:i+1])
        trainpredict = reg.predict(Xtrain[0:i+1])
        errorTrain[i] = np.sum((trainpredict - Ytrain[0:i+1])**2)/len(Ytrain[0:i+1])
        testpredict = reg.predict(Xtest)
        errorTest[i] = np.sum((testpredict - Ytest)**2)/len(Ytest)

    return errorTrain, errorTest
