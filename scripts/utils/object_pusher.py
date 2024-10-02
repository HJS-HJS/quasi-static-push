from sympy import Matrix, MatrixSymbol, zeros, sqrt, simplify, symbols, rot_axis3, pi
import numpy as np
from utils.object_circle import ObjectCircle
from utils.diagram import *

class ObjectPusher(object):
    '''
    2d pusher
    '''
    def __init__(self, n_finger, radius, distance, heading, center_x, center_y, rotation):
        self.pushers = []

        self.q = np.array([center_x, center_y, rotation])
        self.v = np.array([0, 0, 0])

        self.m_q_set = zeros(n_finger, 3)
        self.m_v_set = zeros(n_finger, 3)

        self.sym_q  = MatrixSymbol('qp_', 1, 3)
        self.sym_v  = MatrixSymbol('vp_', 1, 3)
        m_q         = Matrix(self.sym_q)
        m_v         = Matrix(self.sym_v)
        self.radius = radius
        _rot = rot_axis3(m_q[0,2])

        _w = Matrix([[0, 0, m_v[2]]])

        for i in range(n_finger):
            _rel1 = distance * np.cos(2 * np.pi / n_finger * i + heading)
            _rel2 = distance * np.sin(2 * np.pi / n_finger * i + heading)

            if np.abs(_rel1) < 1e-10: _rel1 = 0
            if np.abs(_rel2) < 1e-10: _rel2 = 0

            m_q_rel = Matrix([[_rel1,
                        _rel2,
                        2 * pi / n_finger * i,
                        ]])
            
            _finger_q = zeros(1, 3)
            _finger_v = zeros(1, 3)

            _finger_q[0,:2] = m_q[0,:2] + m_q_rel[0,:2] * _rot[:2,:2]
            _finger_q[0,2]  = m_q[0,2] + m_q_rel[0,2]

            _finger_v[0,:2] = m_v[0,:2] + _w.cross(m_q_rel[0,:] * _rot)[0,:2]
            _finger_v[0,2]  = m_v[0,2]

            _obj = ObjectCircle(_finger_q, _finger_v, radius, False)

            self.pushers.append(_obj)
            self.m_q_set[i,:] = _finger_q
            self.m_v_set[i,:] = _finger_v

        self.m_q_set = simplify(self.m_q_set)
        self.m_v_set = simplify(self.m_v_set)

    def __len__(self): return len(self.pushers)
    
    def __getitem__(self, i): return self.pushers[i]

    def __iter__(self):
        for pusher in self.pushers:
            yield pusher

    def apply_c(self, center):
        self.q[:2] = center

    def apply_rot(self, rotation):
        self.q[2] = rotation

    def apply_v(self, velocity):
        self.v = np.array(velocity)

    def move_q(self, dq):
        self.q += dq

    @property
    def r(self):
        return np.hstack([pusher.radius for pusher in self.pushers])
    
    @property
    def c(self):
        return self.q[:2]

    @property
    def rot(self):
        return self.q[2]

    