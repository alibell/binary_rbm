import numpy as np

def sigmoid(X):
    """sigmoid

    Compute the sigmoid function

    Parameters
    ----------
    X: numpy array

    Output:
    -------
    Numpy array of the same size of X
    """

    return 1/(1+np.exp(-X))

class binary_RBM ():
    def __init__ (self, q, max_iter=300, batch_size=64, stop_criterion=1e-4, lr=0.01):
        """__init__

        Initialisation of the binary_RBM class
        
        Parameters
        ----------
        q: int, number of hidden features
        stop_criterion: float, amplitude of absence of loss changes during 3 iterations to stop the training
        batch_size: int, number of sample to process in each batch
        max_iter: int, maximum number of iteration during training

        lr: float, learning rate
        """
        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            raise ValueError("batch_size should be of type float")

        if isinstance(lr, float):
            self.lr = lr
        else:
            raise ValueError("lr should be of type float")

        if isinstance(stop_criterion, float):
            self.stop_criterion = stop_criterion
        else:
            raise ValueError("stop_criterion should be of type float")

        if isinstance(q, int):
            self.q = q
        else:
            raise ValueError("q should be of type int")

        if isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            raise ValueError("max_iter should be of type int")

        self.coefs_ = {
            'a':None,
            'b':None,
            'W':None
        }

    def _init_params(self, X):
        """_init_params
        
        Initialize the network parameters according to the dataset

        Parameters
        ----------
        X: size (n,p) with n the number of samples and p the number of features
        """

        # Initialising the coefs
        self.p_ = X.shape[1]
        self.coefs_["a"] = np.zeros((1,self.p_,))
        self.coefs_["b"] = np.zeros((1,self.q,))
        self.coefs_["W"] = (np.random.randn(self.p_*self.q)*np.sqrt(1e-2)).reshape((self.p_, self.q))

        # Loss list
        self.loss = []

    def _get_conditional_probability(self, variable, variable_value):
        """_get_conditional_probability

            Compute the conditional probability of hidden of visible variable

            Parameters
            ----------
            variable: str, hidden or visible
            variable_value: np vector or size (p) or (q) according to if it is visible or hidden
        """

        if variable not in ("hidden","visible"):
            raise ValueError("Variable should be valued 'hidden' or 'visible'")

        if variable=='hidden':
            Z_ = variable_value.dot(self.coefs_["W"])+self.coefs_["b"]
        else:
            Z_ = variable_value.dot(self.coefs_["W"].T)+self.coefs_["a"]

        res = sigmoid(Z_)

        return res

    def get_hidden(self, X):
        """Get hidden variables

        Use gibbs to sample hidden variable from visible variable
        
        Parameters
        ----------
        X: size (n,p) with n the number of samples and p the number of features

        Output
        ------
        Y: size (n,q) with n the number of sample and q the number of hidden variables 
        """

        hidden_probs = self._get_conditional_probability("hidden", X)
        H = (np.random.rand(X.shape[0],self.q) <= hidden_probs).astype("int")

        return H

    def get_visible(self, H):
        """Get visible variables

        Use gibbs to sample visibles variable from hidden variable
        
        Parameters
        ----------
        X: size (n,q) with n the number of samples and q the number of hidden variables

        Output
        ------
        Y: size (n,p) with n the number of sample and q the number of visibles variables 
        """

        visible_probs = self._get_conditional_probability("visible", H)
        X = (np.random.rand(H.shape[0], self.p_) <= visible_probs).astype("int")

        return X

    def grad(self, X):
        """grad

        Compute the gradient of the parameters

        Parameters
        ----------
        X:  size (n,p) with n the number of samples and p the number of features

        Output
        ------
        Dict containing the gradients of W, a and b
        """

        # Getting X_1, the gibbs estimation of the expectancy p(v=1/h)
        H = self.get_hidden(X)
        X_1 = self.get_visible(H)

        # Getting the probabilities
        h_prob = self._get_conditional_probability("hidden", X)
        h_1_prob = self._get_conditional_probability("hidden", X_1)
        
        # Getting the gradient
        W_grad = (1/X.shape[0])*(X.T.dot(h_prob)-X_1.T.dot(h_1_prob))
        a_grad = (X-X_1).mean(axis=0)
        b_grad = (h_prob-h_1_prob).mean(axis=0)

        # Returning gradients
        grad_dict = {
            "W":W_grad,
            "a":a_grad,
            "b":b_grad
        }

        return grad_dict

    def get_loss(self, X):
        """
            Compute the loss which is the mean square error of X reconstitution

            Parameters
            ----------
            X: size (n,p) with n the number of samples and p the number of features
        """
        h_prob = self._get_conditional_probability("hidden", X)
        x_prob = self._get_conditional_probability("visible", h_prob)

        loss = np.square(X-x_prob).mean()

        return loss

    def fit(self, X, y=None):
        """fit

        Train the binary RBM

        Parameters
        ----------
        X: size (n,p) with n the number of samples and p the number of features
        y: Not expected, keeped for standard fit api compatibility
        """

        # Initialisation of the parameters
        self._init_params(X)
        n_samples = X.shape[0]
        n_batchs = (n_samples//self.batch_size)+int(n_samples%self.batch_size != 0)

        # Keep a record of the number of iter with no changes
        n_iter_no_changes = 0
        for i in range(self.max_iter):
            # We shuffle X
            X_ = X.copy()
            np.random.shuffle(X_)

            for batch in range(n_batchs):

                X_batch_ = X_[batch*self.batch_size:(batch+1)*self.batch_size]

                # Gradient descent step
                grads = self.grad(X_batch_)

                # Perform gradient descent steps
                self.coefs_["W"] += self.lr*grads["W"]
                self.coefs_["a"] += self.lr*grads["a"]
                self.coefs_["b"] += self.lr*grads["b"]

            # Getting the loss
            loss = self.get_loss(X)
            self.loss.append(loss)

            if (len(self.loss) > 1):
                if self.loss[-2]-self.loss[-1] <= self.stop_criterion:
                    n_iter_no_changes = n_iter_no_changes+1
                    if n_iter_no_changes >= 20:
                        return None
                else:
                    n_iter_no_changes = 0
