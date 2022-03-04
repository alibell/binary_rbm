from . import RBM
import numpy as np

class binary_DBN ():
        def __init__ (self, hidden_states=(), max_iter=300, batch_size=64, stop_criterion=1e-4, lr=0.01):
            """
                DBN for binary data
                The DBN is trained as a stack of RBM with p hidden states
                When p == 0, h correspond to the visible variables

                Parameters
                ----------
                hidden_states: tuples for hidden states size
                max_iter: maximum number of iteration for each RBM train
                batch_size: batch_size for RBM train
                stop_criterion: stop criterion for RBM train
                lr: learning rate for RBM train
            """

            # Storing the parameters
            self.params_ = locals()

            # Initializing RBMs
            self.layers_ = []
            for x in self.params_["hidden_states"]:
                self.layers_.append(
                    RBM.binary_RBM(
                        q=x, 
                        max_iter=self.params_["max_iter"], 
                        batch_size=self.params_["batch_size"],
                        stop_criterion=self.params_["stop_criterion"],
                        lr=self.params_["lr"]
                    )
                )

        def fit(self, X, y=None):
            """fit

            Train the binary DBN

            Parameters
            ----------
            X: size (n,p) with n the number of samples and p the number of features
            y: Not expected, keeped for standard fit api compatibility
            """

            # Initializing the training with the X variables
            X_ = X

            # Training each layer
            for i in range(len(self.layers_)):
                self.layers_[i].fit(X_)
                X_ = self.layers_[i]._get_conditional_probability("hidden", X_)

        def get_hidden(self, X):
            """Get hidden variables

                Use the network to sample hidden variables from visible variable
                
                Parameters
                ----------
                X: size (n,p) with n the number of samples and p the number of features

                Output
                ------
                List of hidden variables of size (n, q_p) with q_p the hidden dimension for each layer
            """

            X_ = X
            hidden_variables_proba = []

            for i in range(len(self.layers_)):
                hidden_probs_ = self.layers_[i]._get_conditional_probability("hidden", X_)

                hidden_variables_proba.append(
                    hidden_probs_
                )

                X_ = hidden_variables_proba[-1]

            return hidden_variables_proba

        def generate(self, n_sample=1, p=0.5, n_gibbs=1):
            """generate
            
            Generate data by gibbs sampling according to a binomial distribution on the last layer
            Then propagate to the first layer

            Parameters:
            ----------
            n_sample: int, size of the sample to generate
            p: float, binomial parameter
            n_gibbs: integer, number of gibbs sampling to proceed
            """

            # Gibbs sampling on the last layer
            H = self.layers_[-1].generate(
                n_sample=n_sample,
                p=p,
                n_gibbs=n_gibbs
            )

            # Propagate informations
            for i in range(len(self.layers_)-2, -1, -1):
                H = self.layers_[i]._get_conditional_probability("visible", H)

            # Sampling
            H = (np.random.rand(*H.shape) <= H).astype("int")

            return H