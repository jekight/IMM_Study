import sys
import numpy as np
from scipy.stats import norm, multivariate_normal

class KalmanFilters:
    def __init__(self,model=1,dt=10,s=np.array([]),meas_noise =0.2,process_noise=0.5):
        # Define Kalman filter model
        self.model = model
        # Define sampling time
        self.dt = dt
        # Define initial state
        self.X = s
        # Initial Covariance w/ uncertainty
        self.P = np.eye(len(s)) * 1e-6
        #Measurement Noise
        self.R = np.eye(3) * meas_noise
        # Process Noise
        self.Q = np.eye(int(len(s))) * process_noise
        # Residual
        self.S = np.zeros(3)
        # System uncertainty
        self.S_cov = np.eye(3)
        self.initialize_models()
        # Identity Matrix
        self.I = np.eye(len(s))

    def initialize_models(self):
        # Near Constant Velocity Model
        if self.model == 1:
            # Define Measurement Mapping Matrix
            self.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0],[0,0,1,0,0,0]])
            # Define the State Transition Matrix F
            self.F = np.array([
                [1,0,0,self.dt,0,0],
                [0,1,0,0,self.dt,0],
                [0,0,1,0,0,self.dt],
                [0,0,0,1,0,0], 
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]
            ])
            return
        # Constant Acceleration
        elif self.model == 2:
            # Define Measurement Mapping Matrix
            self.H = np.array([[1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]])
            # Define the State Transition Matrix F
            self.F = np.array([
                [1,0,0,self.dt,0,0,(0.5*self.dt**2),0,0],
                [0,1,0,0,self.dt,0,0,(0.5*self.dt**2),0],
                [0,0,1,0,0,self.dt,0,0,(0.5*self.dt**2)],
                [0,0,0,1,0,0,self.dt,0,0], 
                [0,0,0,0,1,0,0,self.dt,0],
                [0,0,0,0,0,1,0,0,self.dt],
                [0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,1],
            ])
            return
        # Hovering Model
        if self.model == 3:
            # Define Measurement Mapping Matrix
            self.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0],[0,0,1,0,0,0]])
            # Define the State Transition Matrix F
            self.F = np.array([
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,1,0,0,0],
                [0,0,0,1,0,0], 
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]
            ])
            return
        # Hovering Model
        if self.model == 4:
            # Define Measurement Mapping Matrix
            self.H = np.array([[1,0,0], [0,1,0],[0,0,1]])
            # Define the State Transition Matrix F
            self.F = np.array([
                [1,0,0],
                [0,1,0],
                [0,0,1],
            ])
            return
        else:
            self.H = np.array([])
            self.F = np.array([])
            return

    def predict(self):
        # (Eq. 15)
        self.X = np.dot(self.F,self.X)
        # (Eq. 7)
        self.P = np.dot(np.dot(self.F,self.P),self.F.T) + self.Q

    def update(self,measurement):
        # Residual
        self.S = measurement - np.dot(self.H,self.X)
        # Residual uncertainty
        self.S_cov = np.dot(self.H,np.dot(self.P,self.H.T)) + self.R
        # Ensure S_cov is positive definite by adding a small value to the diagonal if necessary
        epsilon = 1e-8
        self.S_cov = self.S_cov + epsilon * np.eye(self.S.shape[0])
        # Compute Kalman Gain
        G = np.dot(np.dot(self.P,self.H.T),np.linalg.inv(self.S_cov))
        # Scale Kalman gain if condition of matrix gets too large
        condition_threshold = 1e3  # Example threshold for condition number
        if np.linalg.cond(self.S_cov) > condition_threshold:
            gain_factor = 0.5  # Scale down gain when condition number is high
            G = gain_factor * G
        # Update State
        self.X += np.dot(G,self.S)
        # Update uncertainties
        I_KH = self.I - np.dot(G,self.H)
        self.P = np.dot(np.dot(I_KH,self.P),I_KH.T) + np.dot(np.dot(G,self.R),G.T)
        # Compute Likelihood
        self.compute_likelihood()

    def compute_likelihood(self):
        # Dimensionality of the residual vector S and epsilon
        k = self.S.shape[0]
        epsilon = 1e-8
        
        # Determinant and inverse of the covariance matrix
        try:
            S_cov_det = np.linalg.det(self.S_cov)
            S_cov_inv = np.linalg.inv(self.S_cov)
        except np.linalg.LinAlgError:
            S_cov_det = np.finfo(float).eps  # Small number to avoid divide-by-zero errors
            S_cov_inv = np.eye(self.S.shape[0])  # Fallback to identity matrix if S_cov is singular
        
        # Calculate the exponent term: -0.5 * S.T * S_cov_inv * S
        exponent_term = -0.5 * np.dot(self.S.T, np.dot(S_cov_inv, self.S))
        
        # Calculate the normalization term
        normalization_term = 1.0 / np.sqrt((2 * np.pi) ** k * max(S_cov_det, epsilon))
        
        # Calculate the likelihood
        self.L = normalization_term * np.exp(exponent_term)
        
        # Avoid zero likelihood by returning a very small value if likelihood is zero
        if self.L == 0:
            self.L = sys.float_info.min

    def compute_likelihood_alt(self):
        # Determinant and inverse of the covariance matrix
        try:
            S_cov_inv = np.linalg.inv(self.S_cov)
        except np.linalg.LinAlgError:
            S_cov_inv = np.eye(self.S.shape[0])  # Fallback to identity matrix if S_cov is singular
        
        # # Calculate Likelihood
        # self.L = np.dot(1/np.sqrt(2*np.pi*self.S),(np.exp(-0.5 * np.dot(self.S.T, np.dot(S_cov_inv, self.S)))))

        # # Avoid zero likelihood by returning a very small value if likelihood is zero
        # if self.L == 0:
        #     self.L = sys.float_info.min
        flat_x = np.asarray(self.S).flatten()
        L = multivariate_normal.logpdf(flat_x,None,S_cov_inv)
        self.L = np.exp(L)
        if self.L == 0:
            self.L = 1e-8

    def get_X(self):
        if self.model == 1:
            return self.X 
        elif self.model == 2:
            return self.X[:6]
        elif self.model == 4:
            return np.concatenate((self.X, np.array([0,0,0])))
        else:
            return self.X[:6]
        
    def get_P(self):
        if self.model == 1:
            return self.P 
        elif self.model == 2:
            return self.P[:6,:6]
        elif self.model == 4:
            return np.pad(self.P,((0,3),(0,3)), mode='constant')
        else:
            return self.P[:6,:6]

    def set_X(self,new_x):
        if len(self.X) == 3:
            self.X = new_x[:3]
        if len(self.X) == 6:
            self.X = new_x 
        else:
            self.X[:6] = new_x

    def set_P(self,new_p):
        if len(self.P) == 6:
            self.P = new_p 
        else:
            self.P[:6,:6] = new_p

    def get_uncertainties(self):
        return np.diagonal(self.P)

    def get_meas_noise(self):
        return np.diagonal(self.R)