import numpy as np

class IMM:
    def __init__(self):
        self.mu = np.array([0.75,0.25])
        self.trans_prob = 0.95
        self.mixing_prob =np.array([
            [self.trans_prob  * self.mu[0], (1 - self.trans_prob ) * self.mu[1]],
            [(1 - self.trans_prob ) * self.mu[0], self.trans_prob  * self.mu[1]]
        ])

    def mix_state(self,x_cv=np.array([]),x_ca=np.array([])):
        self.x_mixed = self.mixing_prob[0,0] * x_cv + self.mixing_prob[1,1] * x_ca[:6]

    def mix_covariance(self,p_cv=np.array([]),p_ca=np.array([])):
        # Only mix position & velocity
        self.p_mixed = self.mixing_prob[0,0] * p_cv + self.mixing_prob[1,1] * p_ca[:6,:6]

    def update_probabilities(self,Z=np.array([]),kf_cv=None,kf_ca=None):
        # Calculate residuals for each model
        # r_cv = Z - np.dot(kf_cv.H,kf_cv.X)
        # r_ca = Z - np.dot(kf_ca.H,kf_ca.X)
        # Calculate Likelihoods
        # likelihood_cv = (1 / np.sqrt(np.linalg.det(2 * np.pi * kf_cv.S_cov))) * np.exp(-0.5 * np.dot(r_cv.T, np.dot(np.linalg.inv(kf_cv.S_cov), r_cv)))
        # likelihood_ca = (1 / np.sqrt(np.linalg.det(2 * np.pi * kf_ca.S_cov))) * np.exp(-0.5 * np.dot(r_ca.T, np.dot(np.linalg.inv(kf_ca.S_cov), r_ca)))        # Update Probabilities
        # mu_upd = np.array([self.mu[0] * likelihood_cv, self.mu[1] * likelihood_ca])
        # mu_upd /= np.sum(mu_upd)  # Normalize
        # self.mu = mu_upd

        # likelihood_cv = np.exp(-0.5 * np.linalg.norm(Z - kf_cv.H @ kf_cv.X)**2)
        # likelihood_ca = np.exp(-0.5 * np.linalg.norm(Z - kf_ca.H @ kf_ca.X)**2)
        # mu_upd = np.array([self.mu[0] * likelihood_cv, self.mu[1] * likelihood_ca])
        # mu_upd = np.maximum(mu_upd, 0.01) 
        # mu_upd /= np.sum(mu_upd)  # Normalize
        # self.mu = mu_upd
        # Calculate the residuals (innovations)
        residual_cv = Z - np.dot(kf_cv.H, kf_cv.X)
        residual_ca = Z - np.dot(kf_ca.H, kf_ca.X)
        
        # Measurement prediction covariance for each model
        S_cv = np.dot(np.dot(kf_cv.H, kf_cv.P), kf_cv.H.T) + kf_cv.R
        S_ca = np.dot(np.dot(kf_ca.H, kf_ca.P), kf_ca.H.T) + kf_ca.R

        # Avoid singularities by adding a small epsilon to the determinant
        epsilon = 1e-10
        
        # Calculate the Gaussian likelihood for each model
        likelihood_cv = np.exp(-0.5 * np.dot(residual_cv.T, np.dot(np.linalg.inv(S_cv + epsilon), residual_cv))) \
                        / (np.sqrt((2 * np.pi)**len(Z) * np.linalg.det(S_cv + epsilon)) + epsilon)
        
        likelihood_ca = np.exp(-0.5 * np.dot(residual_ca.T, np.dot(np.linalg.inv(S_ca + epsilon), residual_ca))) \
                        / (np.sqrt((2 * np.pi)**len(Z) * np.linalg.det(S_ca + epsilon)) + epsilon)

        # Update probabilities with normalized likelihoods
        mu_upd = np.array([self.mu[0] * likelihood_cv, self.mu[1] * likelihood_ca])
        
        # Introduce a small floor to prevent probabilities from going to zero
        mu_upd = np.maximum(mu_upd, 0.01)
        self.mu = mu_upd / np.sum(mu_upd)  # Normalize to sum to 1

