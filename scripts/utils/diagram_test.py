import numpy as np
import torch
# from scipy.optimize import minimize
# from sympy import symbols, Matrix, Abs, cos, sin, ones, MatrixSymbol, sqrt, lambdify
from sympy import symbols, Matrix, Abs, cos, sin, ones, MatrixSymbol
from pygame.transform import rotate as pygamerotate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Diagram():
    def __init__(self, x, y, phi):
        self.x = x
        self.y = y
        self.r = ones(1)
        self.phi = phi

        self.sym_t = symbols('t', real=True)

        self.m_fun = Matrix([
            self.x + cos(self.sym_t),
            self.y + sin(self.sym_t)
        ])
        self.initialize()

        self.polygon = None
        self.init_angle = None
        self.torch_points = None

    def initialize(self):
        self.m_grad = self.m_fun.diff(self.sym_t)
        self.gen_torch_pts()

    def points(self, q):
        t = np.linspace(0, np.pi * 2, 100)
        return np.array([self.m_fun.subs({self.sym_t: t_i, MatrixSymbol('qp_', 1, 3):Matrix(q.reshape(1,3))}) for t_i in t]).astype(np.float64).T.reshape(2, -1)
    
    def get_ellipse_pts(self, npts:int=300, tmin:float=0, trange:float=2*np.pi):
        t = np.linspace(tmin, tmin + trange, npts)
        return np.array([self.m_fun.subs({self.sym_t: t_i - self.phi}) for t_i in t]).astype(np.float64).T.reshape(2, npts)
        
    def gen_torch_pts(self, npts:int=100, tmin:float=0, trange:float=2*np.pi):
        self.torch_points = torch.tensor(self.get_ellipse_pts(npts, tmin, trange), device=device, dtype=torch.float32).T

    def get_ellipse_pt(self, angle):
        return np.array(self.m_fun.subs({self.sym_t: angle - self.phi})).astype(np.float64).reshape(-1)
    
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

    @property
    def center(self):
        return np.array([self.x, self.y])
    
    @staticmethod
    def collision_angle(diagram1: "Diagram", diagram2: "Diagram"):

        def angle(v):
            return np.arctan2(v[1], v[0])
        
        distances = torch.cdist(diagram1.torch_points, diagram2.torch_points)

        arg = torch.argmin(distances)

        return [
            angle(diagram1.torch_points[arg // len(distances)].cpu().numpy() - diagram1.center),
            angle(diagram2.torch_points[arg % len(distances)].cpu().numpy() - diagram2.center)
        ]
###
# class CollisionPoint():
#     def __init__(self, obs1:Diagram, obs2:Diagram):
#         self.obs1 = obs1
#         self.obs2 = obs2

#     def solve(self):
#         c_vector = ellipse2.center - ellipse1.center
#         c_angle  = np.arctan2(c_vector[1], c_vector[0])

#         initial_guess = [c_angle, c_angle + np.pi]

#         # dist_func = self.distance_fun()
#         # # 미분(최적화를 위한 기울기 계산)
#         # theta1 = symbols('t_1', real=True)
#         # theta2 = symbols('t_2', real=True)
#         # grad_theta1 = dist_func.diff(theta1)
#         # grad_theta2 = dist_func.diff(theta2)

#         # # lambdify를 통해 SymPy 식을 NumPy/PyTorch로 변환
#         # distance_func = lambdify((theta1, theta2), dist_func, 'numpy')
#         # self.grad_func_theta1 = lambdify((theta1, theta2), grad_theta1, 'numpy')
#         # self.grad_func_theta2 = lambdify((theta1, theta2), grad_theta2, 'numpy')

#         # now = time.time()
#         # theta1 = torch.tensor(c_angle, device=device, requires_grad=True, dtype=torch.float32)
#         # theta2 = torch.tensor(c_angle + torch.pi, device=device, requires_grad=True, dtype=torch.float32)


#         # # PyTorch에서 SGD를 이용한 최적화
#         # optimizer = torch.optim.SGD([theta1, theta2], lr=0.1)

#         # # 최적화 루프
#         # for step in range(100):
#         #     optimizer.zero_grad()  # 그래디언트 초기화
            
#         #     # 거리 계산
#         #     dist = self.distance_torch(distance_func, theta1, theta2)
            
#         #     # 직접 기울기를 계산해 backward()에 반영
#         #     grad = self.grad_torch(theta1, theta2)
#         #     theta1.grad = grad[0]  # theta1에 대한 기울기
#         #     theta2.grad = grad[1]  # theta2에 대한 기울기
            
#         #     optimizer.step()  # 파라미터 업데이트
            
#         #     if step % 100 == 0:
#         #         print(f"Step {step}, Distance: {dist.item()}")

#         # return [theta1, theta2]


#         # solution = minimize(self.distance, initial_guess, bounds=[(c_angle-np.pi/2, c_angle+np.pi/2), (c_angle + np.pi/2, c_angle + np.pi*3/2)])
#         # solution = minimize(self.distance, initial_guess)
#         solution = minimize(self.distance, initial_guess, method="BFGS", tol=1e-1)

#         if solution.success is True:
#             return solution.x
#         else:
#             print("Cant find shorest path")
#             return False

#     def distance(self, params):
#         angle1, angle2 = params
#         return np.linalg.norm(self.obs1.get_ellipse_pt(angle1) - self.obs2.get_ellipse_pt(angle2))
    
#     def distance_fun(self):
#         return (sqrt((self.obs1.m_fun.subs({symbols('t', real=True):symbols('t_1', real=True)}) - self.obs2.m_fun.subs({symbols('t', real=True):symbols('t_2', real=True)})).dot(self.obs1.m_fun.subs({symbols('t', real=True):symbols('t_1', real=True)}) - self.obs2.m_fun.subs({symbols('t', real=True):symbols('t_2', real=True)}))))
    
#     def distance_torch(self, dist_func, theta1, theta2):
#         # NumPy 배열을 PyTorch로 변환하여 계산
#         dist = dist_func(theta1.item(), theta2.item())
#         return torch.tensor(dist, device=device, dtype=torch.float32)

#     def grad_torch(self, theta1, theta2):
#         # NumPy에서 미분 값 구한 후 PyTorch로 변환
#         grad1 = self.grad_func_theta1(theta1.item(), theta2.item())
#         grad2 = self.grad_func_theta2(theta1.item(), theta2.item())
#         return torch.tensor([grad1, grad2], device=device, dtype=torch.float32)
###
class Circle(Diagram):
    def __init__(self, x, y, r, phi):
        super().__init__(x, y, phi)
        self.r = ones(1)[0] * r

        self.m_fun = Matrix([
            self.x + self.r*cos(self.sym_t + self.phi),
            self.y + self.r*sin(self.sym_t + self.phi)
        ])
        self.initialize()

class SuperEllipse(Diagram):
    def __init__(self, x, y, a, b, n, phi):
        super().__init__(x, y, phi)
        self.a   = a
        self.b   = b
        
        self.r = (Abs(cos(self.sym_t) / self.a)**n + Abs(sin(self.sym_t) / self.b)**n)**(-1/n)

        self.m_fun = Matrix([
            self.x + self.r*cos(self.sym_t + self.phi),
            self.y + self.r*sin(self.sym_t + self.phi)
        ])
        self.initialize()

class Ellipse(Diagram):
    def __init__(self, x, y, a, b, phi):
        super().__init__(x, y, phi)
        self.a   = a
        self.b   = b
        
        self.r = ((cos(self.sym_t) / self.a)**2 + (sin(self.sym_t)**2 / self.b))**(-1/2)

        self.m_fun = Matrix([
            self.x + self.r*cos(self.sym_t + self.phi),
            self.y + self.r*sin(self.sym_t + self.phi)
        ])
        self.initialize()


def plot_sol(sol, ellipse1, ellipse2):
    p_ellipse1_2 = ellipse1.get_ellipse_pt(sol[0])
    p_ellipse2_1 = ellipse2.get_ellipse_pt(sol[1])
    _vec1 = ellipse1.normal_vector(sol[0]) / 30
    _vec2 = ellipse2.normal_vector(sol[1]) / 30
    plt.plot([p_ellipse1_2[0], p_ellipse1_2[0] + _vec1[0]], [p_ellipse1_2[1], p_ellipse1_2[1] + _vec1[1]], label="normal line 1")
    plt.plot([p_ellipse2_1[0], p_ellipse2_1[0] + _vec2[0]], [p_ellipse2_1[1], p_ellipse2_1[1] + _vec2[1]], label="normal line 2")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    ellipse1 = SuperEllipse(1.0, 1.0, 0.5, 1.0, 20, np.pi / 6)
    # ellipse2 = SuperEllipse(3.0, 3.0, 2.4, 1.2, 20, -np.pi / 18)
    # ellipse2 = Ellipse(3.0, 3.0, 2.4, 1.2, -np.pi / 18)
    ellipse2 = Circle(3.0, 3.0, 1.2, -np.pi / 18)
    ellipse3 = Ellipse(6.5, 2.0, 3.0, 0.8, np.pi / 9)


    c_vector = ellipse1.center - ellipse2.center
    c_vector = c_vector / np.linalg.norm(c_vector)
    c_angle  = np.arctan2(c_vector[1], c_vector[0])

    ps_ellipse1 = ellipse1.get_ellipse_pts()
    ps_ellipse2 = ellipse2.get_ellipse_pts()
    ps_ellipse3 = ellipse3.get_ellipse_pts()

    ellipse1.gen_torch_pts()
    ellipse2.gen_torch_pts()
    ellipse3.gen_torch_pts()

    plt.figure(figsize=(6, 6))
    plt.plot(ps_ellipse1[0], ps_ellipse1[1], label="Skewed Ellipse 1")
    plt.plot(ps_ellipse2[0], ps_ellipse2[1], label="Skewed Ellipse 2")
    plt.plot(ps_ellipse3[0], ps_ellipse3[1], label="Skewed Ellipse 3")
    plt.scatter(ellipse1.center[0], ellipse1.center[1])
    plt.scatter(ellipse2.center[0], ellipse2.center[1])
    plt.scatter(ellipse3.center[0], ellipse3.center[1])

    plt.plot([ellipse1.center[0], ellipse2.center[0]], [ellipse1.center[1], ellipse2.center[1]], label="center line")
    
    # now0 = time.time()
    # sol = CollisionPoint(ellipse1, ellipse2).solve()
    # print("solution:\t", sol)
    # print("time:\t{:.10f}".format(time.time() - now0))
    
    # now = time.time()
    # sol2 = CollisionPoint(ellipse1, ellipse3).solve()
    # print("solution:\t", sol2)
    # print("time:\t{:.10f}".format(time.time() - now))
    
    # now = time.time()
    # sol3 = CollisionPoint(ellipse2, ellipse3).solve()
    # print("solution:\t", sol3)
    # print("time:\t{:.10f}".format(time.time() - now))

    now = time.time()
    sol4 = Diagram.collision_angle(ellipse1, ellipse2)
    print("solution4:\t", sol4)
    print("time:\t{:.10f}".format(time.time() - now))

    now = time.time()
    sol4 = Diagram.collision_angle(ellipse1, ellipse2)
    print("solution4:\t", sol4)
    print("time:\t{:.10f}".format(time.time() - now))

    now = time.time()
    sol5 = Diagram.collision_angle(ellipse1, ellipse3)
    print("solution4:\t", sol5)
    print("time:\t{:.10f}".format(time.time() - now))

    now = time.time()
    sol6 = Diagram.collision_angle(ellipse2, ellipse3)
    print("solution4:\t", sol6)
    print("time:\t{:.10f}".format(time.time() - now))

    plot_sol(sol4, ellipse1, ellipse2)
    plot_sol(sol5, ellipse1, ellipse3)
    plot_sol(sol6, ellipse2, ellipse3)

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Skewed Ellipse with Rotation")
    plt.show()
