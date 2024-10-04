from sympy import MatrixSymbol, symbols, ones, Matrix, cos, sin
from pygame.transform import rotate as pygamerotate
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Diagram(object):
    '''
    2d object
    '''
    def __init__(self):
        self.radius = 0
        self.q = 0
        self.v = 0
        self.parent_q = 0

        self.r = ones(1)[0] * 0

        self.sym_t = symbols('t', real=True)

        self.m_fun = Matrix([
            0,
            0
        ])

    def initialize(self, q):
        self.polygon = None
        self.init_angle = None
        self.torch_points = None
        self.gen_torch_pts()
        self.m_fun[0,0] += q[0]
        self.m_fun[1,0] += q[1]
        self.m_grad = self.m_fun.diff(self.sym_t)
              
    def gen_torch_pts(self):
        t = np.linspace(0, np.pi * 2, 100)
        self.torch_points = torch.tensor(np.array([self.m_fun.subs({self.sym_t: t_i}) for t_i in t]).astype(np.float64).T.reshape(2, -1), device=device, dtype=torch.float32).T

    def point(self, q, angle):
        return np.array(self.m_fun.subs({self.sym_t: angle, MatrixSymbol('qp_', 1, 3):Matrix(q.reshape(1,3))})).astype(np.float64).T.reshape(2, -1)

    def points(self, q):
        t = np.linspace(0, np.pi * 2, 100)
        return np.array([self.m_fun.subs({self.sym_t: t_i, MatrixSymbol('qp_', 1, 3):Matrix(q.reshape(1,3))}) for t_i in t]).astype(np.float64).T.reshape(2, -1)

    def tangent_vector(self, angle):
        _vec = np.array(self.m_grad.subs({self.sym_t: angle - self.phi})).astype(np.float64).reshape(-1)
        return _vec / np.linalg.norm(_vec)
    
    def normal_vector(self, angle):
        _vec = self.tangent_vector(angle)
        return np.array([_vec[1], -_vec[0]])

    def surface(self, center):
        rotated_triangle = pygamerotate(self.polygon, int(np.rad2deg(center[2] - self.init_angle)))
        polygon_rect = rotated_triangle.get_rect(center=(center[0], center[1]))
        return rotated_triangle, polygon_rect.topleft
    
class Circle(Diagram):
    def __init__(self, q, v, radius, parents_q = None):
        self.radius = radius
        self.q = q
        self.v = v
        self.phi = 0

        if parents_q is None:
            q = MatrixSymbol('qp_', 1, 3)
            self.parent_q = self.q
        else:
            self.parent_q = parents_q
            
        self.r = ones(1)[0] * radius
        
        self.sym_t = symbols('t', real=True)

        self.m_fun = Matrix([
            self.r*cos(self.sym_t + self.phi),
            self.r*sin(self.sym_t + self.phi)
        ])
        
        self.initialize(q)