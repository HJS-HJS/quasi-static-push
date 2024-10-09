import numpy as np
import torch
from pygame.transform import rotate as pygamerotate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Diagram():
    def __init__(self, x, y, phi):
        self.x = x
        self.y = y
        self.r = 1
        self.phi = phi

        self.initialize()

    def func_radius(self, theta):
        return 1

    def func_diagram(self, theta):
        _r = self.func_radius(theta = theta)
        return [
            self.x + _r*np.cos(theta),
            self.y + _r*np.sin(theta)
        ]
    
    def func_gradient(self, theta):
        _r = self.func_radius(theta = theta)
        return [
            - _r*np.sin(theta),
            - _r*np.cos(theta)
        ]
    
    def initialize(self):
        self.polygon = None
        self.init_angle = None
        self.torch_points = None
        self.q = np.zeros(3)
        self.q[0] = self.x
        self.q[1] = self.y
        self.q[2] = self.phi
        self.gen_torch_pts()
    
    def get_ellipse_pts(self, npts:int=300, tmin:float=0, trange:float=2*np.pi):
        theta = np.linspace(tmin, tmin + trange, npts)
        return self.func_diagram(theta)
        
    def gen_torch_pts(self, npts:int=1000, tmin:float=0, trange:float=2*np.pi):
        x_ = self.x
        y_ = self.y
        phi_ = self.phi

        self.x = 0
        self.y = 0
        self.phi = 0
        
        self.torch_points = torch.tensor(self.get_ellipse_pts(npts, tmin, trange), device=device, dtype=torch.float32).T
        self.x = x_
        self.y = y_
        self.phi = phi_

    def get_ellipse_pt(self, theta):
        return self.func_diagram(theta)
    
    def tangent_vector(self, theta):
        _vec = self.func_gradient(theta)
        return _vec / np.linalg.norm(_vec)
    
    def normal_vector(self, theta):
        _vec = self.tangent_vector(theta)
        return np.array([_vec[1], -_vec[0]])

    def surface(self, center):
        rotated_triangle = pygamerotate(self.polygon, int(np.rad2deg(center[2] - self.init_angle)))
        polygon_rect = rotated_triangle.get_rect(center=(center[0], center[1]))
        return rotated_triangle, polygon_rect.topleft

    @property
    def center(self):
        return np.array([self.x, self.y])
    
    @staticmethod
    def collision_angle(diagram1: "Diagram", diagram2: "Diagram"):

        def angle(v):
            return np.arctan2(v[1], v[0])
        
        q1 = torch.tensor(diagram1.q, device=device, dtype= torch.float32)
        q2 = torch.tensor(diagram2.q, device=device, dtype= torch.float32)

        rot_1 = torch.tensor([[torch.cos(q1[2]), -torch.sin(q1[2])],
                              [torch.sin(q1[2]),  torch.cos(q1[2])]], device=device, dtype= torch.float32)
        rot_2 = torch.tensor([[torch.cos(q2[2]), -torch.sin(q2[2])],
                              [torch.sin(q2[2]),  torch.cos(q2[2])]], device=device, dtype= torch.float32)

        distances = torch.cdist(diagram1.torch_points @ rot_1.T + q1[:2], diagram2.torch_points @ rot_2.T + q2[:2])

        arg = torch.argmin(distances)

        return [
            (torch.pi * 2 * (arg // len(distances)) / len(distances)).cpu().numpy() + diagram1.phi,
            (torch.pi * 2 * (arg  % len(distances)) / len(distances)).cpu().numpy() + diagram2.phi,
        ]
    
    @staticmethod
    def collision_dist(diagram1: "Diagram", diagram2: "Diagram"):

        def angle(v):
            return np.arctan2(v[1], v[0])
        
        q1 = torch.tensor(diagram1.q, device=device, dtype= torch.float32)
        q2 = torch.tensor(diagram2.q, device=device, dtype= torch.float32)

        rot_1 = torch.tensor([[torch.cos(q1[2]), -torch.sin(q1[2])],
                              [torch.sin(q1[2]),  torch.cos(q1[2])]], device=device, dtype= torch.float32)
        rot_2 = torch.tensor([[torch.cos(q2[2]), -torch.sin(q2[2])],
                              [torch.sin(q2[2]),  torch.cos(q2[2])]], device=device, dtype= torch.float32)

        torch_points_1 = diagram1.torch_points @ rot_1.T
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

class Circle(Diagram):
    def __init__(self, x, y, r, phi):
        self.x = x
        self.y = y
        self.r = r
        self.phi = phi
        
        self.initialize()

    def func_radius(self, theta):
        return self.r

    def func_diagram(self, theta):
        _r = self.func_radius(theta = theta - self.phi)
        return [
            self.x + _r*np.cos(theta),
            self.y + _r*np.sin(theta)
        ]
    
    def func_gradient(self, theta):
        _r = self.func_radius(theta = theta - self.phi)
        _dr = 0
        return [
            - _r*np.sin(theta) + _dr * np.cos(theta),
            + _r*np.cos(theta) + _dr * np.sin(theta)
        ]
class SuperEllipse(Diagram):
    def __init__(self, x, y, a, b, n, phi):
        self.x = x
        self.y = y
        self.phi = phi
        self.a   = a
        self.b   = b
        self.n   = n
        
        self.initialize()

    def func_radius(self, theta):
        return (np.abs(np.cos(theta) / self.a)**self.n + np.abs(np.sin(theta) / self.b)**self.n)**(-1/self.n)
    
    def func_radius_d(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        term1 = (np.abs(cos_theta / self.a) ** self.n)
        term2 = (np.abs(sin_theta / self.b) ** self.n)
        g_theta = term1 + term2
        
        dg_dtheta = -self.n * (np.abs(cos_theta / self.a) ** (self.n - 1)) * (sin_theta / self.a) + \
                    self.n * (np.abs(sin_theta / self.b) ** (self.n - 1)) * (cos_theta / self.b)
        
        df_dtheta = -(1 / self.n) * g_theta ** (-1 / self.n - 1) * dg_dtheta
        
        return df_dtheta * np.sign(theta)
        # return df_dtheta

    def func_diagram(self, theta):
        _r = self.func_radius(theta = theta - self.phi)
        return np.array([
            self.x + _r*np.cos(theta),
            self.y + _r*np.sin(theta)
        ])
    
    def func_gradient(self, theta):
        _point1 = self.get_ellipse_pt(theta)
        _point2 = self.get_ellipse_pt(theta+0.0001)
        return (_point2 - _point1)
        _r = self.func_radius(theta = theta - self.phi)
        _dr = self.func_radius_d(theta = theta - self.phi)
        return [
            - _r*np.sin(theta) + _dr * np.cos(theta),
            + _r*np.cos(theta) + _dr * np.sin(theta)
        ]

class Ellipse(Diagram):
    def __init__(self, x, y, a, b, phi):
        self.x = x
        self.y = y
        self.phi = phi
        self.a   = a
        self.b   = b

        self.initialize()

    def func_radius(self, theta):
        return ((np.cos(theta) / self.a)**2 + (np.sin(theta)**2 / self.b))**(-1/2)
    
    def func_radius_d(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        g_theta = (cos_theta**2 / self.a**2) + (sin_theta**2 / self.b)
        
        dg_dtheta = -2 * (cos_theta * sin_theta / self.a**2) + 2 * (sin_theta * cos_theta / self.b)
        
        df_dtheta = -0.5 * g_theta ** (-3 / 2) * dg_dtheta
        
        return df_dtheta

    def func_diagram(self, theta):
        _r = self.func_radius(theta = theta - self.phi)
        return [
            self.x + _r*np.cos(theta),
            self.y + _r*np.sin(theta)
        ]
    
    def func_gradient(self, theta):
        _r = self.func_radius(theta = theta - self.phi)
        _dr = self.func_radius_d(theta = theta - self.phi)
        return [
            - _r*np.sin(theta) + _dr * np.cos(theta),
            + _r*np.cos(theta) + _dr * np.sin(theta)
        ]


def plot_sol(ellipse1, ellipse2):
    sol = Diagram.collision_dist(ellipse1, ellipse2)
    p_ellipse1_2 = ellipse1.get_ellipse_pt(sol[0])
    p_ellipse2_1 = ellipse2.get_ellipse_pt(sol[1])
    _vec1 = ellipse1.normal_vector(sol[0]) / 30
    _vec2 = ellipse2.normal_vector(sol[1]) / 30
    plt.plot([p_ellipse1_2[0], p_ellipse1_2[0] + _vec1[0]], [p_ellipse1_2[1], p_ellipse1_2[1] + _vec1[1]], label="normal line 1")
    plt.plot([p_ellipse2_1[0], p_ellipse2_1[0] + _vec2[0]], [p_ellipse2_1[1], p_ellipse2_1[1] + _vec2[1]], label="normal line 2")

def plot_sol_print(ellipse1: Diagram, ellipse2: Diagram):
    sol = Diagram.collision_dist(ellipse1, ellipse2)
    p_ellipse1_2 = ellipse1.get_ellipse_pt(sol[0])
    p_ellipse2_1 = ellipse2.get_ellipse_pt(sol[1])
    print('solutions')
    print(np.rad2deg(sol[0]))
    print(np.rad2deg(sol[1]))
    _vec1 = ellipse1.normal_vector(sol[0]) / 30
    _vec2 = ellipse2.normal_vector(sol[1]) / 30

    _r = ellipse2.func_radius(-sol[1])
    _dr = ellipse2.func_radius_d(-sol[1])
    print("")
    print("sol")
    print("grad:\t", ellipse2.func_gradient(sol[1]))
    print("rad:\t", ellipse2.func_radius(sol[1]))
    print("drad:\t", ellipse2.func_radius_d(sol[1]))
    print("")
    print("-sol")
    print("grad:\t", ellipse2.func_gradient(-sol[1]))
    print("rad:\t", ellipse2.func_radius(-sol[1]))
    print("drad:\t", ellipse2.func_radius_d(-sol[1]))
    print("rad:\t", ellipse2.func_radius(np.pi-sol[1]))
    print("drad:\t", ellipse2.func_radius_d(np.pi-sol[1]))
    print("rad:\t", ellipse2.func_radius(-np.pi+sol[1]))
    print("drad:\t", ellipse2.func_radius_d(-np.pi+sol[1]))

    print('test')
    print(_r*np.sin(sol[1]))
    print(_dr * np.cos(sol[1]))
    print(_r*np.cos(sol[1]))
    print(_dr * np.sin(sol[1]))

    print('test minus')
    print(_r*np.sin(-sol[1]))
    print(_dr * np.cos(-sol[1]))
    print(_r*np.cos(-sol[1]))
    print(_dr * np.sin(-sol[1]))

    plt.plot([p_ellipse1_2[0], p_ellipse1_2[0] + _vec1[0]], [p_ellipse1_2[1], p_ellipse1_2[1] + _vec1[1]], label="normal line 1")
    plt.plot([p_ellipse2_1[0], p_ellipse2_1[0] + _vec2[0]], [p_ellipse2_1[1], p_ellipse2_1[1] + _vec2[1]], label="normal line 2")

    p_ellipse2_1_minus = ellipse2.get_ellipse_pt(-sol[1])
    _vec3 = ellipse2.normal_vector(-sol[1]) / 30
    p_ellipse2_1_180 = ellipse2.get_ellipse_pt(np.pi - sol[1])
    _vec4 = ellipse2.normal_vector(np.pi-sol[1]) / 30
    p_ellipse2_1_m_180 = ellipse2.get_ellipse_pt(-np.pi + sol[1])
    _vec5 = ellipse2.normal_vector(-np.pi+sol[1]) / 30
    plt.plot([p_ellipse2_1_minus[0], p_ellipse2_1_minus[0] + _vec3[0]], [p_ellipse2_1_minus[1], p_ellipse2_1_minus[1] + _vec3[1]], label="normal line 3 minus")
    plt.plot([p_ellipse2_1_180[0], p_ellipse2_1_180[0] + _vec4[0]], [p_ellipse2_1_180[1], p_ellipse2_1_180[1] + _vec4[1]], label="normal line 3 minus")
    plt.plot([p_ellipse2_1_m_180[0], p_ellipse2_1_m_180[0] + _vec5[0]], [p_ellipse2_1_m_180[1], p_ellipse2_1_m_180[1] + _vec5[1]], label="normal line 3 minus")



def plot_diagram(diagram):
    points = diagram.get_ellipse_pts()
    plt.plot(points[0], points[1], label="Skewed Circle")
    plt.scatter(diagram.center[0], diagram.center[1])

def plot_line(diagram1, diagram2):
    plt.plot([diagram1.center[0], diagram2.center[0]], [diagram1.center[1], diagram2.center[1]], label="center line")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    circle = Circle(1.0, 1.0, 1.2, -np.pi / 18)
    ellipse1 = SuperEllipse(2.0, 2.0, 0.5, 1.0, 20, np.pi / 6 + np.pi)
    ellipse2 = SuperEllipse(0.0, -1.0, 1.5, 0.3, 20, 0 + np.pi)
    # ellipse2 = SuperEllipse(0.0, -1.0, 1.5, 0.3, 20, 0)
    ellipse3 = SuperEllipse(-0.5, 0.0, 1.5, 0.3, 20, -np.pi / 4 + np.pi)
    ellipse4 = SuperEllipse(0.8, -0.5, 0.5, 0.5, 20, np.pi / 9 + np.pi)
    # ellipse1 = Ellipse(2.0, 2.0, 0.5, 1.0, 20, np.pi / 6 + np.pi)
    # ellipse2 = Ellipse(0.0, -1.0, 1.5, 0.3, 20, 0 + np.pi)
    # ellipse3 = Ellipse(-0.5, 0.0, 1.5, 0.3, 20, -np.pi / 4 + np.pi)
    # ellipse4 = Ellipse(0.8, -0.5, 0.5, 0.5, 20, np.pi / 9 + np.pi)


    c_vector = ellipse1.center - ellipse2.center
    c_vector = c_vector / np.linalg.norm(c_vector)
    c_angle  = np.arctan2(c_vector[1], c_vector[0])

    ps_circle = circle.get_ellipse_pts()
    ps_ellipse1 = ellipse1.get_ellipse_pts()
    ps_ellipse2 = ellipse2.get_ellipse_pts()
    ps_ellipse3 = ellipse3.get_ellipse_pts()

    # plt.figure(figsize=(6, 6))
    plt.figure()
    plot_diagram(circle)
    plot_diagram(ellipse1)
    plot_diagram(ellipse2)
    plot_diagram(ellipse3)
    plot_diagram(ellipse4)
    
    plot_sol(circle, ellipse1)
    plot_sol_print(circle, ellipse2)
    plot_sol(circle, ellipse3)
    plot_sol(circle, ellipse4)


    # plt.scatter(ellipse1.get_ellipse_pt(sol7[0])[0], ellipse1.get_ellipse_pt(sol7[0])[1], label="temp line test")
    # plt.scatter(ellipse4.get_ellipse_pt(sol7[1])[0], ellipse4.get_ellipse_pt(sol7[1])[1], label="temp line test")
    # plt.scatter(ellipse3.get_ellipse_pt(sol9[0])[0], ellipse3.get_ellipse_pt(sol9[0])[1], label="temp line test")
    # plt.scatter(ellipse5.get_ellipse_pt(sol9[1])[0], ellipse5.get_ellipse_pt(sol9[1])[1], label="temp line test")

    # plot_sol(sol4, ellipse1, ellipse2)
    # plot_sol(sol5, ellipse1, ellipse3)
    # plot_sol(sol6, ellipse2, ellipse3)

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Skewed Ellipse with Rotation")
    plt.show()
