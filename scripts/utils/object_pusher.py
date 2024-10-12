import copy
import numpy as np

from utils.diagram import Diagram, Circle, Ellipse, SuperEllipse, RPolygon, SmoothRPolygon

class ObjectPusher(object):
    '''
    2d pusher
    '''
    def __init__(self, n_finger:int=2, finger_angle:float = 180, type:dict={'type':"circle", 'r': 0.1}, distance:float=1.0, center_x:float=0.0, center_y:float=0.0, rotation:float=0.0):
        
        self.pushers:list[Diagram] = []
        self.q = np.array([center_x, center_y, rotation])
        self.v = np.array([0, 0, 0])

        if   type["type"] == "circle":       _obj = Circle(np.zeros(3),         type['r'])
        elif type["type"] == "ellipse":      _obj = Ellipse(np.zeros(3),        type['a'], type['b'])
        elif type["type"] == "superellipse": _obj = SuperEllipse(np.zeros(3),   type['a'], type['b'], type['n'])
        elif type["type"] == "rpolygon":     _obj = RPolygon(np.zeros(3),       type['a'], type['k'])
        elif type["type"] == "srpolygon":    _obj = SmoothRPolygon(np.zeros(3), type['a'], type['k'])
        
        f_heading = (finger_angle - np.pi) / 2
        self.m_q_rel = np.zeros((n_finger, 3))
        for i in range(n_finger):
            self.m_q_rel[i,:] = np.array([[
                distance * np.cos(-finger_angle * i + f_heading),
                distance * np.sin(-finger_angle * i + f_heading),
                -(finger_angle) / 2 - (np.pi - finger_angle) * i,
                ]])
            
            self.pushers.append(copy.deepcopy(_obj))
        
        self.apply_q(self.q)
        self.apply_v(self.v)

    def __len__(self)->int: return len(self.pushers)
    
    def __getitem__(self, i:int)->Diagram: return self.pushers[i]

    def __iter__(self):
        for pusher in self.pushers:
            yield pusher

    def apply_q(self, q:np.array):
        self.q = np.array(q)
        _rot = self.rot_matrix[:2,:2]
        for idx, pusher in enumerate(self.pushers):
            pusher.q[:2] = self.q[:2] + self.m_q_rel[idx,:2].dot(_rot)
            pusher.q[2]  = self.q[2] + self.m_q_rel[idx,2]

    def apply_v(self, velocity:np.array):
        self.v = np.array(velocity)
        _rot = self.rot_matrix
        _w = np.array([0, 0, self.v[2]])
        for idx, pusher in enumerate(self.pushers):
            pusher.v[:2] = self.v[:2] + np.cross(_w, self.m_q_rel[idx,:].dot(_rot))[:2]
            pusher.v[2]  = self.v[2]
    
    def pusher_dv(self, dt:float=0.001)->np.array:
        d_set = np.tile(np.eye(3) * dt,reps=[len(self), 1]).reshape(len(self), 3, 3)
        _w = np.array([0, 0, dt])
        d_set[:,2,:2] += np.cross(_w, self.m_q_rel[:,:].dot(self.rot_matrix))[:,:2]

        return d_set
    
    @property
    def r(self)->np.array:
        return np.hstack([pusher.radius for pusher in self.pushers])
    
    @property
    def c(self)->np.array:
        return self.q[:2]

    @property
    def rot(self)->np.array:
        return self.q[2]
    
    @property
    def rot_matrix(self)->np.array:
        return np.array([
            [np.cos(self.q[2]),  np.sin(self.q[2]), 0],
            [np.sin(self.q[2]), -np.cos(self.q[2]), 0],
            [0, 0, 1],
        ])
