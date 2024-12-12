import os
import numpy as np
import configparser
from KalmanFilters.kalman_filters import KalmanFilters

class IMM:
    def __init__(self,filters=[],mu=np.array([]),M=np.array([])):
        '''
        IMM Filter
        Author: Jeremy Kight
        '''
        # Set Filters
        self.filters = filters
        # Set States
        self.X = np.zeros(6)
        self.P = np.zeros([6,6])
        # Mode probabilities
        self.mu = mu
        # Transition probabilities matrix
        self.M = M
        # Likelihoods
        self.L = np.zeros(len(filters))
        # Initialize Markov chain
        self.omega = np.zeros([len(filters),len(filters)])
        # Compute mixing probabilities
        self.compute_mixing_probabilities()
        # Mix states
        self.mix_models()
        
    def compute_mixing_probabilities(self):
        # Compute total probability
        epsilon = 1e-6
        self.cbar = np.dot(self.mu,self.M) + epsilon
        for i in range(len(self.mu)):
            for j in range(len(self.mu)):
                self.omega[i,j] = (self.M[i,j]*self.mu[i]) / self.cbar[j]

    def mix_models(self):
        '''
        Computes the IMM's mixed state using mode probabilities to weight 
        the estimates
        '''
        self.X.fill(0)
        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            X = f.get_X()
            self.X += X * mu
        for f, mu in zip(self.filters, self.mu):
            X = f.get_X()
            P = f.get_P()
            y = X - self.X
            self.P += mu * (np.outer(y, y) + P)
        
    def predict(self):
        '''
        Predict next state (prior) using the IMM state propagation equations
        '''
        # compute mixed initial conditions
        Xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = np.zeros(self.X.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.get_X() * wj
            Xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.get_X() - x
                P += wj * (np.outer(y, y) + kf.get_P())
            Ps.append(P)

        #  compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.set_X(Xs[i])
            f.set_P(Ps[i])
            f.predict()

        # compute mixed IMM state and covariance and save posterior estimate
        self.mix_models()

    def update(self,z=np.array([])):
        # run update on each filter, and save the likelihood
        for i, f in enumerate(self.filters):
            f.update(z)
            self.L[i] = f.L

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.L
        self.mu /= np.sum(self.mu)  # normalize
        min_probability = 1e-5  # Small positive value
        self.mu = np.maximum(self.mu, min_probability)
        self.previous_mu = self.mu.copy()

        self.compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self.mix_models()

