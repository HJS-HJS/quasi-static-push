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
        self.parent_q = 0
        self.dt = 0.1

    def initialize(self):
        self.v = np.zeros(3)
        self.polygon = None
        self.init_angle = None
        self.torch_points = None
        self.limit_constant = None
        _q = self.q
        self.q = np.array([0, 0, 0])
        self.gen_torch_points()
        self.gen_limit_constant()
        self.q = _q
              
    def func_radius(self, theta):
        return 1
    def func_radius_d(self, theta):
        return 0

    def func_diagram(self, theta):
        _r = self.func_radius(theta = theta - self.q[2])
        return np.array([
            self.q[0] + _r*np.cos(theta),
            self.q[1] + _r*np.sin(theta)
        ])
    
    def func_gradient(self, theta):
        _r = self.func_radius(theta = theta - self.q[2])
        _dr = self.func_radius_d(theta = theta - self.q[2])
        return np.array([
            - _r*np.sin(theta) + _dr * np.cos(theta),
            + _r*np.cos(theta) + _dr * np.sin(theta)
        ])
        # + np.random.rand(2) * 0.01
        # return self.func_diagram(theta + 0.0001) - self.func_diagram(theta)

    def point(self, theta):
        return self.func_diagram(theta)
    
    def points(self, npts:int=300, tmin:float=0, trange:float=2*np.pi):
        theta = np.linspace(tmin, tmin + trange, npts, endpoint=False)
        return self.func_diagram(theta)

    def gen_torch_points(self, npts:int=1000, tmin:float=0, trange:float=2*np.pi):
        self.torch_points = torch.tensor(self.points(npts, tmin, trange), device=device, dtype=torch.float32).T

    def tangent_vector(self, theta):
        _vec = self.func_gradient(theta)
        return _vec / np.linalg.norm(_vec)
    
    def normal_vector(self, theta):
        _vec = self.tangent_vector(theta)
        return np.array([_vec[1], -_vec[0]])

    def local_velocity(self, theta):
        return self.v[:2] + self.v[2]*self.func_radius(theta=theta)*self.tangent_vector(theta=theta)
    
    def local_velocity_grad(self, theta, dt, dv = None):
        if dv is None:
            # dv = np.tile(self.v,len(_v)).reshape(len(_v),-1) + np.eye(3) * dt
            dv = np.eye(3) * dt
        return dv[:,:2] + np.outer(dv[:,2], self.func_radius(theta=theta)*self.tangent_vector(theta=theta))

    def surface(self, center):
        rotated_triangle = pygamerotate(self.polygon, int(np.rad2deg(center[2] - self.init_angle)))
        polygon_rect = rotated_triangle.get_rect(center=(center[0], center[1]))
        return rotated_triangle, polygon_rect.topleft
    
    def collision_angle(self, diagram2: "Diagram"):

        def angle(v):
            return np.arctan2(v[1], v[0])
        
        q1 = torch.tensor(self.q, device=device, dtype= torch.float32)
        q2 = torch.tensor(diagram2.q, device=device, dtype= torch.float32)

        rot_1 = torch.tensor([[torch.cos(q1[2]), -torch.sin(q1[2])],
                              [torch.sin(q1[2]),  torch.cos(q1[2])]], device=device, dtype= torch.float32)
        rot_2 = torch.tensor([[torch.cos(q2[2]), -torch.sin(q2[2])],
                              [torch.sin(q2[2]),  torch.cos(q2[2])]], device=device, dtype= torch.float32)

        torch_points_1 = self.torch_points @ rot_1.T
        torch_points_2 = diagram2.torch_points @ rot_2.T + q2[:2] - q1[:2]
        
        diagram1_vector = torch_points_1[1:] - torch_points_1[:-1]  # (99, 2)
        points_2_expanded = torch_points_2.unsqueeze(1)  # (10, 1, 2)
        test_vectors = points_2_expanded - torch_points_1[:-1]  # (10, 99, 2)
        cross_products = torch.sign((diagram1_vector.unsqueeze(0)[:, :, 0] * test_vectors[:, :, 1] - 
                                     diagram1_vector.unsqueeze(0)[:, :, 1] * test_vectors[:, :, 0]))  # (10, 99)

        is_overlap = torch.all(cross_products >= 0, dim=1)
        if not torch.any(is_overlap):
            distances = torch.cdist(torch_points_1, torch_points_2)

            arg = torch.argmin(distances)
            return [
                angle(torch_points_1[arg // len(distances)].cpu().numpy()),
                angle((torch_points_2[arg % len(distances)] - q2[:2] + q1[:2]).cpu().numpy()),
                distances[arg // len(distances), arg % len(distances)].cpu().numpy()
            ]
        else:
            inside_points = torch_points_2[is_overlap]

            distances = torch.cdist(torch_points_1, inside_points)
            _min = torch.min(distances, dim=0)
            diagram2_arg = torch.argmax(_min[0])
            diagram1_arg = _min[1][diagram2_arg]

            dist = _min[0][diagram2_arg]

            return [
                angle(torch_points_1[diagram1_arg].cpu().numpy()),
                angle((inside_points[diagram2_arg] - q2[:2] + q1[:2]).cpu().numpy()),
                -(dist).cpu().numpy()
            ]
    
    def gen_limit_constant(self):
        _npts = 1000
        _dtheta = np.pi * 2 / _npts
        _line = torch.linspace(start=0, end=2*torch.pi, steps = _npts, device=device)
        _r = torch.norm(self.torch_points, dim=1)
        # Full surface
        _A = (1 / 2) * torch.sum(_r ** 2).cpu().numpy() * _dtheta
        # _M = (2 / 9) * torch.sum(_r ** 3).cpu().numpy() * _dtheta + (2 / 9) * torch.sum(_r ** 3).cpu().numpy() * _dtheta
        _M = (2 / 9) * torch.sum(_r ** 3).cpu().numpy() * _dtheta
        # along the edge
        # _A = 2 * torch.sum(_r).cpu().numpy() * _dtheta
        _M = (1 / 2) * torch.sum(_r ** 2).cpu().numpy() * _dtheta * 50
        self.limit_constant = np.array([
            [_A, 0, 0],
            [0, _A, 0],
            [0, 0, _M],
            ])

class Circle(Diagram):
    def __init__(self, q, radius):
        self.q = np.array(q)
        self.radius = radius

        self.initialize()

    def func_radius(self, theta):
        return self.radius
    
    @property
    def r(self):
        return self.radius

class SuperEllipse(Diagram):
    def __init__(self, q, a, b, n):
        self.q = np.array(q)
        self.a = a
        self.b = b
        self.n = n
        
        self.initialize()

    def func_radius(self, theta):
        return (np.abs(np.cos(theta) / self.a)**self.n + np.abs(np.sin(theta) / self.b)**self.n)**(-1/self.n)
    
    def func_radius_d(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        term1 = (np.abs(cos_theta / self.a) ** self.n)
        term2 = (np.abs(sin_theta / self.b) ** self.n)
        g_theta = (-1 / self.n) * (term1 + term2) ** (-1 / self.n - 1)
        
        dg_dtheta = self.n * (np.abs(cos_theta / self.a) ** (self.n - 1)) * (-sin_theta / self.a) * np.sign(cos_theta) + \
                    self.n * (np.abs(sin_theta / self.b) ** (self.n - 1)) * ( cos_theta / self.b) * np.sign(sin_theta)
        
        df_dtheta = g_theta * dg_dtheta
        
        return df_dtheta

    @property
    def r(self):
        return np.sqrt(np.power(self.a, 2) + np.power(self.b, 2))

class Ellipse(Diagram):
    def __init__(self, q, a, b):
        self.q = np.array(q)
        self.a = a
        self.b = b

        self.initialize()

    def func_radius(self, theta):
        return ((np.cos(theta) / self.a)**2 + (np.sin(theta) / self.b)**2)**(-1/2)
    
    def func_radius_d(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        g_theta = (cos_theta / self.a)**2 + (sin_theta / self.b)**2
        
        dg_dtheta = -2 * (cos_theta * sin_theta / self.a**2) + 2 * (sin_theta * cos_theta / self.b**2)
        
        df_dtheta = -0.5 * g_theta ** (-3 / 2) * dg_dtheta
        
        return df_dtheta

    @property
    def r(self):
        return np.sqrt(np.power(self.a, 2) + np.power(self.b, 2))