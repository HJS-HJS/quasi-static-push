import math
import numpy as np
from utils.diagram import Diagram
from utils.object_obstacle import ObjectObstacle
from utils.object_slider   import ObjectSlider
from utils.object_pusher   import ObjectPusher

class ParamFunction(object):
    '''
    Calculate objects param
    '''
    def __init__(self, 
                 sliders:ObjectSlider, 
                 pushers:ObjectPusher,
                 obstacles:ObjectObstacle, 
                 threshold:float = 5e-3,
                 fmscale:  float = 0.2,
                 fascale:  float = 0.9,
                 fbscale:  float = 0.001,
                 ):
        self.sliders   = sliders
        self.pushers   = pushers
        self.obstacles = obstacles
        self.threshold = threshold

        self.fmscale = fmscale
        self.fascale = fascale
        self.fbscale = fbscale

        self.n_phi  = len(self.pushers) * len(self.sliders)\
              + ParamFunction.combination(len(self.sliders), 2)\
              + (len(self.pushers) + len(self.sliders)) * len(self.obstacles)

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
        self.vc_jac = np.zeros((self.n_phi * 2, len(self.q)))
        self.JN     = np.zeros((self.n_phi,     len(self.q)))
        self.JT     = np.zeros((2 * self.n_phi, len(self.q)))
        
        # delta t for vc jacobian
        _dt = 0.0001

        pusher_dv = self.pushers.pusher_dv(_dt)
        # print(pusher_dv)

        i = -1
        n_slider = len(self.sliders)
        # Parameter calculation between pusher and slider
        for i_s, slider in enumerate(self.sliders):
            for i_p, pusher in enumerate(self.pushers):
                i += 1
                # Near diagram check
                if i_s != 0: 
                    if not self.is_collision_available(slider, pusher, self.threshold): continue
                ans = slider.cal_collision_data(pusher)
                # check collision distance
                self.phi[i]    = ans[2]
                self.nhat[i,:] = slider.normal_vector(ans[0])
                # self.nhat[i,:] = -pusher.normal_vector(ans[1])
                self.vc_jac[2*i:2*i+2,3*i_s:3*i_s+3] =           -slider.local_velocity_grad(ans[0], _dt).T / _dt
                self.vc_jac[2*i:2*i+2,3*n_slider:3*n_slider+4] =  pusher.local_velocity_grad(ans[1], _dt, pusher_dv[i_p]).T / _dt

        # Parameter calculation between sliders
        for i_s1 in range(len(self.sliders)):
            for i_s2 in range(i_s1 + 1, len(self.sliders)):
                i += 1
                if not self.is_collision_available(self.sliders[i_s1], self.sliders[i_s2], self.threshold): continue
                ans = self.sliders[i_s1].cal_collision_data(self.sliders[i_s2])
                # check collision distance
                self.phi[i]    = ans[2]
                self.nhat[i,:] = self.sliders[i_s1].normal_vector(ans[0])
                self.vc_jac[2*i:2*i+2,3*i_s1:3*i_s1+3] = -self.sliders[i_s1].local_velocity_grad(ans[0], _dt).T / _dt
                self.vc_jac[2*i:2*i+2,3*i_s2:3*i_s2+3] =  self.sliders[i_s2].local_velocity_grad(ans[1], _dt).T / _dt

        # Parameter calculation between slider and obstacle
        for obs in self.obstacles:
            for idx, diagram in enumerate(self.sliders):
                i += 1
                if not self.is_collision_available(diagram, obs, self.threshold): continue
                ans = diagram.cal_collision_data(obs)
                # check collision distance
                self.phi[i]    = ans[2]
                self.nhat[i,:] = diagram.normal_vector(ans[0])
                self.vc_jac[2*i:2*i+2,3*idx:3*idx+3] = -diagram.local_velocity_grad(ans[0], _dt).T / _dt

        # Parameter calculation between pusher and obstacle
        for obs in self.obstacles:
            for diagram in self.pushers:
                i += 1
                if not self.is_collision_available(diagram, obs, self.threshold): continue
                ans = diagram.cal_collision_data(obs)
                # check collision distance
                self.phi[i]    = ans[2]
                self.nhat[i,:] = diagram.normal_vector(ans[0])
                self.vc_jac[2*i:2*i+2,3*n_slider:3*n_slider+3] = -diagram.local_velocity_grad(ans[0], _dt).T / _dt

        # Update jacobian
        _rot = np.array([[0, -1], [1, 0]])
        for i in range(self.n_phi):
            self.JN[i,:]       =  self.nhat[i,:].dot(self.vc_jac[2*i:2*i+2,:])
            self.JT[2*i,:]     = -(_rot.dot(self.nhat[i,:])).dot(self.vc_jac[2*i:2*i+2,:])
            self.JT[2*i + 1,:] =  (_rot.dot(self.nhat[i,:])).dot(self.vc_jac[2*i:2*i+2,:])

        self.JNS = self.JN[:,:3 * n_slider]
        self.JNP = self.JN[:,3 * n_slider:]
        self.JTS = self.JT[:,:3 * n_slider]
        self.JTP = self.JT[:,3 * n_slider:]

        # sim matrix
        self.mu = np.eye(self.n_phi) * self.fmscale
        self.A  = np.zeros((3 * n_slider, 3 * n_slider))
        self.B  = np.eye(len(self.pushers.q)) * self.fbscale
        self.B[2:,2:] *= 0.01

        for idx, slider in enumerate(self.sliders):
            self.A[3*idx:3*idx + 3,3*idx:3*idx + 3] = slider.limit_constant * self.fascale

    def get_simulate_param(self):
        _thres_idx = np.where(self.phi < 10)
        # _thres_idx = np.where(self.phi < self.threshold / 5)
        _thres_idx_twice = np.repeat(_thres_idx,2) * 2
        _thres_idx_twice[::2] += 1
        return self.qs, \
               self.qp, \
               self.phi[_thres_idx], \
               self.JNS[_thres_idx], \
               self.JNP[_thres_idx], \
               self.JTS[_thres_idx_twice], \
               self.JTP[_thres_idx_twice], \
               self.mu[:len(_thres_idx[0]),:len(_thres_idx[0])],\
               self.A,\
               self.B
    
    @property
    def q(self) -> np.array:
        return np.hstack([self.sliders.q, self.pushers.q])

    @property
    def qs(self) -> np.array:
        return self.sliders.q.reshape(-1)
    
    @property
    def qp(self) -> np.array:
        return self.pushers.q
    
    @property
    def v(self) -> np.array:
        return np.hstack([self.sliders.v, self.pushers.v])
    
    @staticmethod
    def is_collision_available(diagram1:Diagram, diagram2:Diagram, threshold:float) -> bool:
        if (np.linalg.norm((diagram1.q - diagram2.q)[:2]) - diagram1.r - diagram2.r) < threshold: return True
        return False

    @staticmethod
    def combination(n:int, r:int) -> int:
        if n < r:
            return 0
        else:
            return int(math.factorial(n) / (math.factorial(n - r) * math.factorial(r)))