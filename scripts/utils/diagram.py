from pygame.transform import rotate as pygamerotate
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Diagram(object):
    '''
    2d object
    '''
    def __init__(self):
        self.q:np.array = np.zeros(3)
        self.radius:float = 0
        self.dt:float = 0.1

    def initialize(self):
        self.v:np.array = np.zeros(3)
        self.polygon = None
        self.torch_points:torch.tensor = None
        self.limit_constant:np.array = None
        self.gen_torch_points()
        self.gen_limit_constant()
              
    def func_radius(self, theta:np.array) -> np.array:
        return 1
    def func_radius_d(self, theta:np.array) -> np.array:
        return 0

    def func_diagram(self, theta:np.array) -> np.array:
        _r = self.func_radius(theta = theta - self.q[2])
        return np.array([
            self.q[0] + _r*np.cos(theta),
            self.q[1] + _r*np.sin(theta)
        ])
    
    def func_gradient(self, theta:np.array) -> np.array:
        _r = self.func_radius(theta = theta - self.q[2])
        _dr = self.func_radius_d(theta = theta - self.q[2])
        return np.array([
            - _r*np.sin(theta) + _dr * np.cos(theta),
            + _r*np.cos(theta) + _dr * np.sin(theta)
        ])

    def point(self, theta:np.array) -> np.array:
        return self.func_diagram(theta)
    
    def points(self, npts:int=300, tmin:float=0, trange:float=2*np.pi) -> np.array:
        theta = np.linspace(tmin, tmin + trange, npts, endpoint=False)
        return self.func_diagram(theta)

    def gen_torch_points(self, npts:int=1000, tmin:float=0, trange:float=2*np.pi):
        _q = self.q
        self.q = np.array([0, 0, 0])
        self.torch_points = torch.tensor(self.points(npts, tmin, trange), device=device, dtype=torch.float32).T
        self.q = _q

    def tangent_vector(self, theta:np.array) -> np.array:
        _vec = self.func_gradient(theta)
        return _vec / np.linalg.norm(_vec)
    
    def normal_vector(self, theta:np.array) -> np.array:
        _vec = self.tangent_vector(theta)
        return np.array([_vec[1], -_vec[0]])

    def local_velocity(self, theta:np.array) -> np.array:
        return self.v[:2] + self.v[2]*self.func_radius(theta=theta)*self.tangent_vector(theta=theta)
    
    def local_velocity_grad(self, theta:np.array, dt:float = 0.001, dv:np.array = None) -> np.array:
        if dv is None:
            dv = np.eye(3) * dt
        return dv[:,:2] + np.outer(dv[:,2], self.func_radius(theta=theta)*self.rot_vector(theta=theta+np.pi/2))

    def surface(self, center:np.array):
        rotated_triangle = pygamerotate(self.polygon, int(np.rad2deg(center[2])))
        polygon_rect = rotated_triangle.get_rect(center=(center[0], center[1]))
        return rotated_triangle, polygon_rect.topleft
    
    def cal_collision_data(self, diagram2: "Diagram"):

        def angle(v:torch.tensor) -> float:
            return torch.atan2(v[1], v[0]).cpu().numpy()
        
        q1 = torch.tensor(self.q, device=device, dtype= torch.float32)
        q2 = torch.tensor(diagram2.q, device=device, dtype= torch.float32)

        rot_1 = torch.tensor([[torch.cos(q1[2]), -torch.sin(q1[2])],
                              [torch.sin(q1[2]),  torch.cos(q1[2])]], device=device, dtype= torch.float32)
        rot_2 = torch.tensor([[torch.cos(q2[2]), -torch.sin(q2[2])],
                              [torch.sin(q2[2]),  torch.cos(q2[2])]], device=device, dtype= torch.float32)

        diagram1_points = self.torch_points @ rot_1.T
        diagram2_points = diagram2.torch_points @ rot_2.T + q2[:2] - q1[:2]
        
        # If the sign does not change by cross producting the tangent vector of diagram 1 and the point of diagram 2, the point of diagram 2 exists inside diagram 1.
        # Tangent vector of diagram 1
        diagram1_tangent_vectors = diagram1_points[1:] - diagram1_points[:-1]  # (99, 2)
        # Point vector of diagram 2
        diagram2_checker_vectors = diagram2_points.unsqueeze(1) - diagram1_points[:-1]      # (100, 99, 2)
        # cross product two vector sets
        cross_products = torch.sign((diagram1_tangent_vectors.unsqueeze(0)[:, :, 0] * diagram2_checker_vectors[:, :, 1] - 
                                     diagram1_tangent_vectors.unsqueeze(0)[:, :, 1] * diagram2_checker_vectors[:, :, 0])) # (10, 99)

        # Check if there is an external product whose sign does not change
        is_overlap = torch.all(cross_products >= 0, dim=1)

        # If overlap not exists
        if not torch.any(is_overlap):
            # Calculate the shortest distance between each point
            distances = torch.cdist(diagram1_points, diagram2_points)
            arg = torch.argmin(distances)
            return [
                angle(diagram1_points[arg // len(distances)]),
                angle(diagram2_points[arg % len(distances)] - q2[:2] + q1[:2]),
                distances[arg // len(distances), arg % len(distances)].cpu().numpy()
            ]
        # If overlap exists
        else:
            # Calculate distance from internal overlapping points
            # Get overlapping points
            inside_points = diagram2_points[is_overlap]

            distances = torch.cdist(diagram1_points, inside_points)
            # Calculate shortest distance from each internal overlapping points
            _min = torch.min(distances, dim=0)
            # Calculate longest distance from each shortest distances
            diagram2_arg = torch.argmax(_min[0])
            diagram1_arg = _min[1][diagram2_arg]

            return [
                angle(diagram1_points[diagram1_arg]),
                angle(inside_points[diagram2_arg] - q2[:2] + q1[:2]),
                -(_min[0][diagram2_arg]).cpu().numpy()
            ]
    
    def gen_limit_constant(self):
        _npts = 1000
        _dtheta = np.pi * 2 / _npts
        _thickness = 0.005
        _r = torch.norm(self.torch_points, dim=1)

        # Full surface
        # _A = (1 / 2) * torch.sum(_r ** 2).cpu().numpy() * _dtheta
        # _M = (1 / 2) * torch.sum(_r ** 2).cpu().numpy() * _dtheta * 50
        # Along the edge
        _A = (1 / 2) * torch.sum(2 * _thickness*_r - _thickness**2).cpu().numpy() * _dtheta
        _M = torch.sum((1 / 3) * _r ** 3 - (_thickness / 4) * _r**2).cpu().numpy() * _dtheta
        self.limit_constant = np.array([
            [_A, 0, 0],
            [0, _A, 0],
            [0, 0, _M],
            ])

    def rot_vector(self, theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
        ])
    
    def __del__(self):
        del self.v
        del self.polygon
        del self.torch_points
        del self.limit_constant
        torch.cuda.empty_cache() 

class Circle(Diagram):
    def __init__(self, q, radius):
        # Property of superellipse
        self.q:np.array   = np.array(q)
        self.radius:float = radius

        # Value for checking collision availability
        self.r:float = radius

        self.initialize()

    def func_radius(self, theta:np.array) -> float:
        return self.radius

class SuperEllipse(Diagram):
    def __init__(self, q, a, b, n):
        # Property of superellipse
        self.q:np.array = np.array(q)
        self.a:float = a
        self.b:float = b
        self.n:int   = n

        # Value for checking collision availability
        self.r = np.sqrt(np.power(self.a, 2) + np.power(self.b, 2))
        
        self.initialize()

    def func_radius(self, theta:np.array):
        return (np.abs(np.cos(theta) / self.a)**self.n + np.abs(np.sin(theta) / self.b)**self.n)**(-1/self.n)
    
    def func_radius_d(self, theta:np.array):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        term1 = (np.abs(cos_theta / self.a) ** self.n)
        term2 = (np.abs(sin_theta / self.b) ** self.n)

        g_theta = (-1 / self.n) * (term1 + term2) ** (-1 / self.n - 1)
        
        dg_dtheta = self.n * (np.abs(cos_theta / self.a) ** (self.n - 1)) * (-sin_theta / self.a) * np.sign(cos_theta) + \
                    self.n * (np.abs(sin_theta / self.b) ** (self.n - 1)) * ( cos_theta / self.b) * np.sign(sin_theta)
        
        df_dtheta = g_theta * dg_dtheta
        
        return df_dtheta

class Ellipse(Diagram):
    def __init__(self, q, a, b):
        self.q:np.array = np.array(q)
        self.a:float = a
        self.b:float = b

        # Value for checking collision availability
        self.r = np.sqrt(np.power(self.a, 2) + np.power(self.b, 2))

        self.initialize()

    def func_radius(self, theta:np.array) -> np.array:
        return ((np.cos(theta) / self.a)**2 + (np.sin(theta) / self.b)**2)**(-1/2)
    
    def func_radius_d(self, theta:np.array) -> np.array:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        g_theta = -0.5 * ((cos_theta / self.a)**2 + (sin_theta / self.b)**2) ** (-3 / 2)
        
        dg_dtheta = -2 * (cos_theta * sin_theta / self.a**2) + 2 * (sin_theta * cos_theta / self.b**2)
        
        df_dtheta = g_theta * dg_dtheta
        
        return df_dtheta

class SmoothRPolygon(Diagram):
    def __init__(self, q, a, k:int):
        self.q:np.array = np.array(q)
        self.a:float = a
        self.b:float = a / k / 5
        self.k:float = k

        # Value for checking collision availability
        self.r = self.a + self.b

        self.initialize()

    def func_radius(self, theta:np.array) -> np.array:
        return self.a + self.b * np.cos(self.k * theta)
    
    def func_radius_d(self, theta:np.array) -> np.array:
        return - self.k * self.b * np.sin(self.k * theta)
    
class RPolygon(Diagram):
    def __init__(self, q, a, k:int):
        self.q:np.array = np.array(q)
        self.a:float = a
        self.n:float = k

        # Value for checking collision availability
        self.r = self.a

        self.initialize()

    def func_radius(self, theta:np.array) -> np.array:
        return np.cos(np.pi/self.n) / np.cos(theta % (2 * np.pi / self.n) - np.pi / self.n) * self.a

    
    def func_radius_d(self, theta:np.array) -> np.array:
        theta_mod = theta % (2 * np.pi / self.n) - np.pi / self.n

        return self.a * np.cos(np.pi / self.n) * np.sin(theta_mod) / (np.cos(theta_mod) ** 2)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def plot_diagram(diagram:Diagram):
        points = diagram.points()
        plt.plot(points[0], points[1], label=str(type(diagram)))
        plt.scatter(diagram.q[0], diagram.q[1])

    def plot_center_line(diagram1:Diagram, diagram2:Diagram):
        plt.plot([diagram1.q[0], diagram2.q[0]], [diagram1.q[1], diagram2.q[1]], label="center line")

    def plot_solution(diagram1:Diagram, diagram2:Diagram):
        sol = diagram1.cal_collision_data(diagram2)
        p_ellipse1_2 = diagram1.point(sol[0])
        p_ellipse2_1 = diagram2.point(sol[1])
        _vec1 = diagram1.normal_vector(sol[0]) / 10
        _vec2 = diagram2.normal_vector(sol[1]) / 10
        plt.plot([p_ellipse1_2[0], p_ellipse1_2[0] + _vec1[0]], [p_ellipse1_2[1], p_ellipse1_2[1] + _vec1[1]], label="normal line 1")
        plt.plot([p_ellipse2_1[0], p_ellipse2_1[0] + _vec2[0]], [p_ellipse2_1[1], p_ellipse2_1[1] + _vec2[1]], label="normal line 2")

    diagram_set = []
    diagram_set.append(Circle([1.0, 1.0, -np.pi / 18], 1.2))
    diagram_set.append(Circle([0.0, 2.0, np.pi / 2], 0.3))
    diagram_set.append(SuperEllipse([2.0, 2.0, np.pi / 9], 1.0, 1.0, 20))
    diagram_set.append(SuperEllipse([-0.2, -0.5, np.pi + np.pi/9], 0.2, 1.0, 20))
    diagram_set.append(Ellipse([0.8, -0.5, 0], 0.3, 0.5))
    diagram_set.append(Ellipse([2.3, -1.0, -np.pi/4], 0.7, 0.5))
    diagram_set.append(RPolygon([-1, -1, 0], 0.6, 3))

    plt.figure()
    for idx_1 in range(len(diagram_set)):
        plot_diagram(diagram_set[idx_1])
        for idx_2 in range(idx_1 + 1, len(diagram_set)):
            plot_solution(diagram_set[idx_1], diagram_set[idx_2])
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Diagram example")
    plt.show()