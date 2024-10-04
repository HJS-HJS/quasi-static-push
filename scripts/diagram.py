import numpy as np
from scipy.optimize import minimize
from sympy import symbols, Matrix, Abs, cos, sin, ones


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

    def initialize(self):
        self.m_grad = self.m_fun.diff(self.sym_t)

    def get_ellipse_pts(self, npts:int=300, tmin:float=0, trange:float=2*np.pi):
        t = np.linspace(tmin, tmin + trange, npts)
        return np.array([self.m_fun.subs({self.sym_t: t_i - self.phi}) for t_i in t]).astype(np.float64).T.reshape(2, npts)
        
    def get_ellipse_pt(self, angle):
        return np.array(self.m_fun.subs({self.sym_t: angle - self.phi})).astype(np.float64).reshape(-1)
    
    def tangent_vector(self, angle):
        _vec = np.array(self.m_grad.subs({self.sym_t: angle - self.phi})).astype(np.float64).reshape(-1)
        return _vec / np.linalg.norm(_vec)
    
    def normal_vector(self, angle):
        _vec = self.tangent_vector(angle)
        return np.array([_vec[1], -_vec[0]])

    @property
    def center(self):
        return np.array([self.x, self.y])

class CollisionPoint():
    def __init__(self, obs1:Diagram, obs2:Diagram):
        self.obs1 = obs1
        self.obs2 = obs2

    def solve(self):
        c_vector = ellipse2.center - ellipse1.center
        c_angle  = np.arctan2(c_vector[1], c_vector[0])

        initial_guess = [c_angle, c_angle + np.pi]

        solution = minimize(self.distance, initial_guess)

        if solution.success is True:
            return solution.x
        else:
            print("Cant find shorest path")
            return False

    def distance(self, params):
        angle1, angle2 = params
        return np.linalg.norm(self.obs1.get_ellipse_pt(angle1) - self.obs2.get_ellipse_pt(angle2))
    
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ellipse1 = SuperEllipse(1.0, 1.0, 0.5, 1.0, 20, np.pi / 6)
    # ellipse2 = SuperEllipse(3.0, 3.0, 2.4, 1.2, 20, -np.pi / 18)
    # ellipse2 = Ellipse(3.0, 3.0, 2.4, 1.2, -np.pi / 18)
    ellipse2 = Circle(3.0, 3.0, 1.2, -np.pi / 18)

    c_vector = ellipse1.center - ellipse2.center
    c_vector = c_vector / np.linalg.norm(c_vector)
    c_angle  = np.arctan2(c_vector[1], c_vector[0])

    ps_ellipse1 = ellipse1.get_ellipse_pts()
    ps_ellipse2 = ellipse2.get_ellipse_pts()

    plt.figure(figsize=(6, 6))
    plt.plot(ps_ellipse1[0], ps_ellipse1[1], label="Skewed Ellipse 1")
    plt.plot(ps_ellipse2[0], ps_ellipse2[1], label="Skewed Ellipse 2")
    plt.scatter(ellipse1.center[0], ellipse1.center[1])
    plt.scatter(ellipse2.center[0], ellipse2.center[1])

    plt.plot([ellipse1.center[0], ellipse2.center[0]], [ellipse1.center[1], ellipse2.center[1]], label="center line")
    p_ellipse1_2 = ellipse1.get_ellipse_pt(np.pi + c_angle)
    p_ellipse2_1 = ellipse2.get_ellipse_pt(c_angle)
    _vec1 = ellipse1.normal_vector(np.pi + c_angle) / 8
    _vec2 = ellipse2.normal_vector(c_angle) / 8
    plt.plot([p_ellipse1_2[0], p_ellipse1_2[0] + _vec1[0]], [p_ellipse1_2[1], p_ellipse1_2[1] + _vec1[1]], label="normal line 1")
    plt.plot([p_ellipse2_1[0], p_ellipse2_1[0] + _vec2[0]], [p_ellipse2_1[1], p_ellipse2_1[1] + _vec2[1]], label="normal line 2")
    sol = CollisionPoint(ellipse1, ellipse2).solve()
    print(sol)
    p_ellipse1_2 = ellipse1.get_ellipse_pt(sol[0])
    p_ellipse2_1 = ellipse2.get_ellipse_pt(sol[1])
    _vec1 = ellipse1.normal_vector(sol[0]) / 30
    _vec2 = ellipse2.normal_vector(sol[1]) / 30
    plt.plot([p_ellipse1_2[0], p_ellipse1_2[0] + _vec1[0]], [p_ellipse1_2[1], p_ellipse1_2[1] + _vec1[1]], label="normal line 1")
    plt.plot([p_ellipse2_1[0], p_ellipse2_1[0] + _vec2[0]], [p_ellipse2_1[1], p_ellipse2_1[1] + _vec2[1]], label="normal line 2")

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Skewed Ellipse with Rotation")
    plt.show()