import numpy as np
import scipy.linalg
import time 

class KalmanFilter3(object):

    def __init__(self, dt,  std_x, std_y) -> None:
        self.t = time.time()
        self._dt = dt
        ndim = 4
        self.x_k = None
        self._F = np.eye(ndim)  # transition matrix
        for i in range(ndim//2):
            self._F[i, i+2] = dt

        std_sigmaX = (self._dt**6)/4
        
        std_sigmaXY = (self._dt**3)/2
        # std_sigmaVX = (self._dt**2)
        self.Q = np.eye(ndim)  # process noise
        for i in range(ndim//2):
            self.Q[i, i] = 500000
        for i in range(2, ndim):
            self.Q[i, i] = 120
        self.P_k = np.array([  # state covariance matrix
            [std_sigmaX**4, std_sigmaXY, 0, 0],
            [std_sigmaXY, std_sigmaX**4, 0, 0],
            [0, 0, 10**8, 0],
            [0, 0, 0,10**8]
        ])
        self.H = np.eye(2, 4)
        self.R = np.array([[std_x**2, 0],
                          [0, std_y**2]])

    def __str__(self):
        print( self.x_k)
        

    def get_bounding_box(self, bounding_box):
        self.x_k = bounding_box
    
    def reinitialize(self, x, y)-> None:
        t = time.time()
        del_t = t - self.t
        
        u = (x - self.x_k[0]) / del_t
        v = (y - self.x_k[1]) / del_t
        self.x_k =  np.array([[x], [y], u, v])

    def predict(self):
        # print(self.x_k)
        self.x_k = self._F @ self.x_k # state predict  
        # print("predict:", self.x_k)
        self.P_k = (self._F @ self.P_k @ self._F.T) + self.Q
        return self.x_k
    
    def update(self, z_k):
        z_k = np.array(z_k)
        inv_ = scipy.linalg.inv(( self.H @ self.P_k @ self.H.T + self.R))
        self.Kalman = (self.P_k @ self.H.T ) @ inv_
        # print("update:", self.x_k)
        self.x_k = self.x_k + self.Kalman @ (z_k  - self.H @ self.x_k)
        self.P_k = self.P_k - self.Kalman @ self.H @ self.P_k
        return self.x_k
        
        
if __name__ == '__main__':
    obj = KalmanFilter3(0.5, 0, 0, 2, 2)
    time.sleep(0.2)
    obj.reinitialize(1, 3)
    obj.__str__()
    obj.predict()
    # for i in range(2):
    #     obj.predict()
    obj.__str__()
    obj.update([[3.34341177], [10.03023531]])
    obj.__str__()