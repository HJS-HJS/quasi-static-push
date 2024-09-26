import numpy as np

class ObjectCircle(object):
    '''
    2d object
    '''
    def __init__(self, radius, center_x, center_y, rotation:float = 0.0):
        self.radius = radius
        self.q = np.array([center_x, center_y, rotation])
        self.v = np.array([0, 0, 0])

    def point_v(self, norm):
        _arr = np.array([[0, -1], [1, 0]])
        return self.v[2] * self.radius * _arr @ norm + self.v[:2]

    def set_c(self, center):
        self.q[:2] = center

    def set_q(self, q):
        self.q = q

    def set_v(self, v):
        self.v = v

    @property
    def r(self):
        return self.radius
    
    @property
    def q_deg(self):
        return np.hstack([self.c, np.rad2deg(self.rot)])
    
    @property
    def c(self):
        return self.q[:2]
        
    @property
    def v_deg(self):
        return np.hstack([self.v[:2], np.rad2deg(self.v[2])])
    
    @property
    def rot(self):
        return self.q[2]