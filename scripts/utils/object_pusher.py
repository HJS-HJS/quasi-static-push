import numpy as np

from utils.diagram import Circle

class ObjectPusher(object):
    '''
    2d pusher
    '''
    def __init__(self, n_finger, radius, distance, heading, center_x, center_y, rotation):
        self.pushers = []

        self.q = np.array([center_x, center_y, rotation])
        self.v = np.array([0, 0, 0])
        self.radius = radius

        self.m_q_rel = np.zeros((n_finger, 3))

        for i in range(n_finger):
            _rel1 = distance * np.cos(2 * np.pi / n_finger * i + heading)
            _rel2 = distance * np.sin(2 * np.pi / n_finger * i + heading)

            self.m_q_rel[i,:] = np.array([[_rel1,
                                           _rel2,
                                           2 * np.pi / n_finger * i,
                                           ]])
            
            _obj = Circle(np.zeros(3), np.zeros(3), radius)

            self.pushers.append(_obj)
        
        self.apply_q(self.q)
        self.apply_v(self.v)

    def __len__(self): return len(self.pushers)
    
    def __getitem__(self, i): return self.pushers[i]

    def __iter__(self):
        for pusher in self.pushers:
            yield pusher

    def apply_q(self, q):
        self.q = np.array(q)
        _rot = self.rot_matrix
        for idx, pusher in enumerate(self.pushers):
            pusher.q[:2] = self.q[:2] + self.m_q_rel[idx,:2].dot(_rot)
            pusher.q[2]  = self.q[2] + self.m_q_rel[idx,2]

    def apply_v(self, velocity):
        self.v = np.array(velocity)
        _rot = np.eye(3)
        _rot[:2,:2] = self.rot_matrix
        _w = np.array([0, 0, self.v[2]])
        for idx, pusher in enumerate(self.pushers):
            pusher.v[:2] = self.v[:2] + np.cross(_w, self.m_q_rel[idx,:].dot(_rot))[:2]
            pusher.v[2]  = self.v[2]

    @property
    def r(self):
        return np.hstack([pusher.radius for pusher in self.pushers])
    
    @property
    def c(self):
        return self.q[:2]

    @property
    def rot(self):
        return self.q[2]
    
    @property
    def rot_matrix(self):
        return np.array([
            [np.cos(self.q[2]),  np.sin(self.q[2])],
            [np.sin(self.q[2]), -np.cos(self.q[2])],
        ])
