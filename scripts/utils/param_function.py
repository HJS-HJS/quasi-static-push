import math
import numpy as np

class ParamFunction(object):
    '''
    Calculate objects param
    '''
    def __init__(self, sliders, pushers, threshold:float = 1e-1):
        self.sliders   = sliders
        self.pushers   = pushers
        self.threshold = threshold

        self.n_phi  = len(self.pushers) * len(self.sliders) + ParamFunction.combination(len(self.sliders), 2)

        self.phi    = np.zeros(self.n_phi)
        self.nhat   = np.zeros((self.n_phi, 2))
        self.vc     = np.zeros((self.n_phi, 2))
        self.vc_jac = np.zeros((self.n_phi * 2,   len(self.v)))

        self.m_JN     = np.zeros((self.n_phi,     len(self.q)))
        self.m_JT     = np.zeros((2 * self.n_phi, len(self.q)))

    def update_param(self):

        # Initialize matrix
        self.phi    = np.zeros(self.n_phi)
        self.nhat   = np.zeros((self.n_phi, 2))
        self.vc     = np.zeros((self.n_phi, 2))
        self.vc_jac = np.zeros((self.n_phi * 2, len(self.v)))
        self.JN     = np.zeros((self.n_phi,     len(self.q)))
        self.JT     = np.zeros((2 * self.n_phi, len(self.q)))
        
        _dt = 0.0001

        pusher_dv = self.pushers.pusher_dv(_dt)

        i = 0
        n_slider = len(self.sliders)
        for i_s, slider in enumerate(self.sliders):
            for i_p, pusher in enumerate(self.pushers):
                ans = slider.collision_angle(pusher)
                # check collision distance
                self.phi[i]    = ans[2]
                self.nhat[i,:] = slider.normal_vector(ans[0])
                self.vc_jac[2*i:2*i+2,3*i_s:3*i_s+3] =           -slider.local_velocity_grad(ans[0], _dt).T / _dt
                self.vc_jac[2*i:2*i+2,3*n_slider:3*n_slider+3] =  pusher.local_velocity_grad(ans[1], _dt, pusher_dv[i_p]).T / _dt
                i += 1
        
        for i_s1 in range(len(self.sliders)):
            for i_s2 in range(i_s1 + 1, len(self.sliders)):
                ans = self.sliders[i_s1].collision_angle(self.sliders[i_s2])
                # check collision distance
                self.phi[i]    = ans[2]
                self.nhat[i,:] = self.sliders[i_s1].normal_vector(ans[0])
                self.vc_jac[2*i:2*i+2,3*i_s1:3*i_s1+3] = -self.sliders[i_s1].local_velocity_grad(ans[0], _dt).T / _dt
                self.vc_jac[2*i:2*i+2,3*i_s2:3*i_s2+3] =  self.sliders[i_s2].local_velocity_grad(ans[1], _dt).T / _dt
                i += 1

        _rot = np.array([[0, -1], [1, 0]])
        for i in range(self.n_phi):
            self.JN[i,:]       =  self.nhat[i,:].dot(self.vc_jac[2*i:2*i+2,:])
            self.JT[2*i,:]     = -(_rot.dot(self.nhat[i,:])).dot(self.vc_jac[2*i:2*i+2,:])
            self.JT[2*i + 1,:] =  (_rot.dot(self.nhat[i,:])).dot(self.vc_jac[2*i:2*i+2,:])

        self.JNS = self.JN[:,:3 * n_slider]
        self.JNP = self.JN[:,3 * n_slider:]
        self.JTS = self.JT[:,:3 * n_slider]
        self.JTP = self.JT[:,3 * n_slider:]

    @property
    def q(self):
        return np.hstack([self.sliders.q, self.pushers.q])

    @property
    def qs(self):
        return self.sliders.q.reshape(-1)
    
    @property
    def qp(self):
        return self.pushers.q
    
    @property
    def v(self):
        return np.hstack([self.sliders.v, self.pushers.v])

    def get_simulate_param(self):
        _thres_idx = np.where(self.phi < self.threshold)
        _thres_idx_twice = np.repeat(_thres_idx,2) * 2
        _thres_idx_twice[::2] += 1

        return self.qs, \
               self.qp, \
               self.phi[_thres_idx], \
               self.JNS[_thres_idx], \
               self.JNP[_thres_idx], \
               self.JTS[_thres_idx_twice], \
               self.JTP[_thres_idx_twice]
    
    @staticmethod
    def combination(n, r):
        if n < r:
            return 0
        else:
            return int(math.factorial(n) / (math.factorial(n - r) * math.factorial(r)))