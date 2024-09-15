from typing import Any, List, Union

import numpy as np
from scipy import sparse


class LogisticRegression:
    def __init__(self) -> None:
        self.w = None
        self.loss_history: Union[List[Any], None] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ) -> "LogisticRegression":
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01  # type: ignore

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices, :]
            y_batch = y[indices]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w -= learning_rate * gradW
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return self

    def predict_proba(self, X: Any, append_bias: bool = False) -> Any:
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the probabilities of classes in y_proba.   #
        # Hint: It might be helpful to use np.vstack and np.sum                   #
        ###########################################################################
        predictions = self.sigmoid(X.dot(self.w.T))  # type: ignore
        y_proba = np.vstack([1 - predictions, predictions]).T
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_proba

    @staticmethod
    def sigmoid(z: Any) -> Any:
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X: np.ndarray) -> Any:
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = y_proba.argmax(axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray, reg: Any) -> tuple:
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        loss = 0
        # Compute loss and gradient. Your code should not contain python loops.

        h = self.sigmoid(X_batch.dot(self.w))  # type: ignore
        loss = -np.dot(y_batch, np.log(h)) - np.dot((1 - y_batch), np.log(1.0 - h))
        dw = (h - y_batch) * X_batch

        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.
        num_train = X_batch.shape[0]
        loss = loss / num_train

        dw = dw / num_train

        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.

        loss += (reg / (2.0 * num_train)) * np.dot(self.w[:-1], self.w[:-1])  # type: ignore
        dw[:-1] = dw[:-1] + (reg * self.w[:-1]) / num_train  # type: ignore

        return loss, dw

    @staticmethod
    def append_biases(X: np.ndarray) -> Any:
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
