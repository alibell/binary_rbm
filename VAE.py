from multiprocessing.sharedctypes import Value
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Multiply, Add, Dense, Input, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.optimizers import Adam

def sampling(args):
    """
        This function provide a sample according to the latent space mean and variance
        
        Parameters:
        -----------
        args: tuple of z_mean and z_log_var, both are tensor of size (n, q) with q the size of the latent space
    """
    
    z_mean, z_log_var = args
    #Reparameterization trick

    batch_size = K.shape(z_mean)[0]
    z_dim = K.shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch_size,z_dim))
    z_sigma = K.exp(0.5 * z_log_var)
    z_epsilon = Multiply()([z_sigma,epsilon])
    z_rand = Add()([z_mean,z_epsilon])

    return z_rand

class binaryVAE():


    def __init__ (self, p, q, hidden_states, lr=1e-2):
        """
            Initialisation of the VAE

            Parameters:
            -----------
            q: size of the latent space
            hiddent_states: tuples of the hidden states
        """

        self.q = q
        self.p = p

        # Building network

        ## Encoder
        input_img = Input(shape=(p,))
        z = Sequential([
            *[Dense(x, activation="relu") for x in hidden_states]
        ])(input_img)
        z_mean = Dense(q, name='z_mean')(z)
        z_log_var = Dense(q, name='z_log_var')(z)
        z_rand = Lambda(sampling, output_shape=(q,))([z_mean, z_log_var])
        self.encoder = Model(input_img, [z_mean, z_log_var, z_rand], name='encoder')

        ## Decoder
        latent_inputs = Input(shape=(q,), name='z_sampling')
        y =  Sequential([
            *[Dense(x, activation="relu") for x in hidden_states]
        ])(latent_inputs)
        output_img = Dense(p, activation='sigmoid')(y)
        self.decoder = Model(latent_inputs, output_img, name='decoder')

        ## Global model
        output_img = self.decoder(self.encoder(input_img)[2])
        self.model = Model(input_img, output_img, name='vae_mlp')
        loss = self._vae_loss(input_img, output_img, z_mean, z_log_var)
        optimizer = Adam(lr)
        self.model.add_loss(loss)
        self.model.compile(optimizer=optimizer)

    def _vae_loss(self, x, y, z_mean, z_log_var):
        """
            Loss function for VAE
            Should be used with Keras

            Parameters:
            -----------
            x: input data of size (n, p)
            y: decoder output image of size (n, p)
            z_mean: latent space mean estimation of size (n, q)
            z_log_var: latent space log of the variance, of size (n, q)

            Outputs:
            --------
            Loss value being the sum of the reconstruction loss and the KL divergence
        """
        reconstruction_loss = K.sum(binary_crossentropy(x, y), axis=-1)
        kl_div = 0.5*K.sum(
            K.exp(z_log_var)+K.square(z_mean)-z_log_var-1
        , axis=-1)
        
        vae_loss = reconstruction_loss+kl_div
        vae_loss = K.mean(vae_loss)

        return vae_loss

    def summary(self):
        """
            Return model summary
        """

        return self.model.summary()

    def fit(self, X, verbose=True, n_epoch=10, batch_size=64, lr=None):
        """
            Train the model

            Parameters:
            -----------
            X: train samples
        """

        if lr is not None:
            K.set_value(self.model.optimizer.learning_rate, lr)

        self.model.fit(X, X, verbose=verbose, epochs=n_epoch, batch_size=batch_size)

        return self
    
    def generate(self, n_images):
        """
            Generate new data

            Parameters:
            -----------
            n_images: int, number of images to generate
        """

        z_sample = np.random.normal(0,1,(n_images,self.q))
        y_hat = self.decoder(z_sample)

        return y_hat