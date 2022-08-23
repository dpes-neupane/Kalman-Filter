from re import I
from matplotlib import projections
import numpy as np
import scipy.linalg

class Kalman(object):

    def __init__(self):
        ndim, dt = 4, 2. # number of dimensions of the measurement and dt = diff between two measurement.    
        # create a 8x8 matrix for update_mean
        self._update_mean = np.eye(2*ndim, 2*ndim)
        # create a F matrix 
        for i in range(ndim):
            self._update_mean[i, ndim+i] = dt

        # create a 4x8 matrix for update_motion can be the observation matrix for projection to measurement space 
        self._update_motion = np.eye(ndim, 2*ndim)
        # uncertainty 
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    
    
    def initialize(self, measurement):
        # create a mean vector using the 4-dim measurement
        mean_pos = np.array(measurement)
        # create a mean vector for velocity (4-dim)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
                2 * self._std_weight_position * measurement[0],
                2 * self._std_weight_position * measurement[1],
                1 * measurement[2],
                2 * self._std_weight_position * measurement[3],
                10 * self._std_weight_velocity * measurement[0],
                10 * self._std_weight_velocity * measurement[1],
                0.1 * measurement[2],
                10 * self._std_weight_velocity * measurement[3],
            ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    
    def predict(self, mean, covariance):
        # use the previous mean and covariance to compute the new mean and covariance 
        std = [
                2 * self._std_weight_position * mean[0],
                2 * self._std_weight_position * mean[1],
                1 * mean[2],
                2 * self._std_weight_position * mean[3],
                2 * self._std_weight_velocity * mean[0],
                2 * self._std_weight_velocity * mean[1],
                0.1 * mean[2],
                2 * self._std_weight_velocity * mean[3]
            ]
        # this the noise matrix Q        
        mean_w_motion = np.diag(np.square(std))
        
        # step 1 :
        # calculate mean(x) = F * mean(last-timestamp)
        mean = np.dot(self._update_mean, mean)
        # step 2:
        # calculate covariance(P) = F P (last-timestamp) F.T + Q
        covariance = np.linalg.multi_dot([ self._update_mean, covariance, self._update_mean.T ] ) + mean_w_motion
        return mean, covariance

    
    def _project(self, mean, covariance):
        '''
        The estimated state distribution should be projected to measurement space.
        '''
        
        
        # initialize
        std = [
                 self._std_weight_position * mean[0],
                 self._std_weight_position * mean[1],
                0.1 * mean[2],
                self._std_weight_position * mean[3],
        ]
        
        # create a R matrix by making a diagonal matrix of the square of the initialized matrix (4 x 4)
        R = np.diag(np.square(std))
        
        # make the 8-dim matrix to 4-dim by performing dot product between mean and a 4x8 matrix (Hx)
        mean = np.dot(self._update_motion, mean)
        

        
        
        # project the covariance matrix to measurement space to obtain HP'H.T. Here, H = 4x8 matrix (observation_matrix)
        covariance = np.linalg.multi_dot([self._update_motion, covariance, self._update_motion.T])
        return mean, covariance+R
    
    
    def update(self, mean, covariance, measurement):
        
        #get the projection into the measurement space
        mean_projected, covariance_projected = self._project(mean, covariance)
        
        # step 1 :
        # calculate Kalman Gain(K) = P(now) H (H P(now) H.T + R)^-1
        cholesky, lower = scipy.linalg.cho_factor(covariance_projected, lower=True, check_finite=False)

        K = scipy.linalg.cho_solve((cholesky, lower), np.dot(covariance, self._update_motion.T).T, check_finite=False).T
        
        # step 2:
        # calculate residual(y) = z(now) - H * mean(now)....... (z = measurement ( 4x1 vector) )
        y = measurement - mean_projected
        
        
        # step 3:
        # calculate the new updated mean = x(now) + Kalman Gain(K) * residual(y)
        new_mean = mean + np.dot(K, y)
        
        
        # step 4:
        # calculate the new covariance (P) = (I - Kalman(K) * Observation Matrix (H)) Covariance (P)
        new_covariance = covariance - np.linalg.multi_dot([K, covariance_projected, K.T])
        return new_mean, new_covariance
        

    
    
if __name__ == "__main__":
    obj = Kalman()
    mean, covariance = obj.initialize([0,10,20, 30])
    print(mean)
    # print(covariance)
    mean, covariance = obj.predict(mean, covariance)
    # mean, covariance = obj._project(mean, covariance)
    mean, covariance = obj.update(mean, covariance, [10,20,20, 35])
    print(mean)
    mean, covariance = obj._project(mean,covariance) 
    print(mean)
    # print(covariance)
