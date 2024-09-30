import math
import numpy as np
from sympy import Matrix, MatrixSymbol, zeros, sqrt, simplify, rot_axis3

class ParamFunction(object):
    '''
    Calculate objects param
    '''
    def __init__(self, sliders, pushers, simplify):
        self.sliders = sliders
        self.pushers = pushers
        self.simplify = simplify

        self.initialize_param()

        # self.show_symbols()

    def show_symbols(self):
        print('')
        print('self.m_qs\n', self.m_qs)
        print('self.m_qp\n', self.m_qp)
        print('self.m_rs\n', self.m_rs)
        print('self.m_rp\n', self.m_rp)
        print('self.m_vs\n', self.m_vs)
        print('self.m_vp\n', self.m_vp)

        print('')
        print('self.m_phi\n',       len(self.m_phi), self.m_phi)
        print('self.m_nhat\n',      len(self.m_nhat), self.m_nhat)
        print('self.m_vc\n',        len(self.m_vc), self.m_vc)
        print('self.m_v_jaco\n',    len(self.m_vc_jac), self.m_vc_jac)

        print('')
        print('self.m_JN', self.m_JN.shape, self.m_JN)
        print('self.m_JT', self.m_JT.shape, self.m_JT)

    def initialize_param(self):
        self.sym_qs = MatrixSymbol('qs_', len(self.sliders), 3)
        self.sym_vs = MatrixSymbol('vs_', len(self.sliders), 3)
        self.sym_rs = MatrixSymbol('rs_', 1, len(self.sliders))
        self.sym_qp = self.pushers.sym_q
        self.sym_vp = self.pushers.sym_v
        self.sym_rp = MatrixSymbol('rp_', 1, len(self.pushers))

        self.m_qs = Matrix(self.sym_qs)
        self.m_vs = Matrix(self.sym_vs)
        self.m_rs = Matrix(self.sym_rs)
        self.m_qp = Matrix(self.sym_qp)
        self.m_vp = Matrix(self.sym_vp)
        self.m_rp = Matrix(self.sym_rp)

        self.m_qp_p = self.pushers.m_q_set
        self.m_vp_p = self.pushers.m_v_set

        self.m_v = Matrix([self.m_vs.col_join(self.m_vp)[:]])

        n_phi = len(self.pushers) * len(self.sliders) + ParamFunction.combination(len(self.sliders), 2)
        
        self.m_phi  = zeros(n_phi, 1)
        self.m_nhat = zeros(n_phi, 2)
        self.m_vc   = zeros(n_phi, 2)

        _rot = Matrix([[0, -1], [1, 0]])

        i = 0
        for i_s in range(len(self.sliders)):
            for i_p in range(len(self.pushers)):
                self.m_phi[i] = ParamFunction.norm(self.m_qp_p[i_p,0:2] - self.m_qs[i_s,0:2]) - self.m_rp[i_p] - self.m_rs[i_s]
                self.m_nhat[i,:] = ParamFunction.unit_vector(self.m_qp_p[i_p,0:2] - self.m_qs[i_s,0:2])
                point_vel = (self.m_vs[i_s,2] * self.m_rs[i_s] * _rot * self.m_nhat[i,:].T).T + self.m_vs[i_s,:2]
                self.m_vc[i,:] = self.m_vp_p[i_p,0:2] - point_vel
                i += 1
        
        for i_s1 in range(len(self.sliders)):
            for i_s2 in range(len(self.sliders) - i_s1 - 1):
                i_s3 = i_s1 + i_s2 + 1
                self.m_phi[i] = ParamFunction.norm(self.m_qs[i_s3,0:2] - self.m_qs[i_s1,0:2]) - self.m_rs[i_s3] - self.m_rs[i_s1]
                self.m_nhat[i,:] = ParamFunction.unit_vector(self.m_qs[i_s3,0:2] - self.m_qs[i_s1,0:2])
                point_vel1 = (self.m_vs[i_s1,2] * self.m_rs[i_s1] * _rot * self.m_nhat[i,:].T).T + self.m_vs[i_s1,:2]
                point_vel3 = -(self.m_vs[i_s3,2] * self.m_rs[i_s3] * _rot * self.m_nhat[i,:].T).T + self.m_vs[i_s3,:2]
                self.m_vc[i,:] = point_vel3 - point_vel1
                i += 1

        self.m_vc_jac = self.m_vc.reshape(1,n_phi * 2).jacobian(self.m_v)

        if self.simplify:
            self.m_phi    = simplify(self.m_phi)
            self.m_nhat   = simplify(self.m_nhat)
            self.m_vc     = simplify(self.m_vc)
            self.m_vc_jac = simplify(self.m_vc_jac)

        self.m_JN     = zeros(n_phi, len(self.q))
        self.m_JT     = zeros(2 * n_phi, len(self.v))

        for i in range(n_phi):
            self.m_JN[i,:] = self.m_nhat[i,:] * self.m_vc_jac[i*2:i*2+2,:]
            self.m_JT[2*i,:] = (_rot * self.m_nhat[i,:].T).T*self.m_vc_jac[i*2:i*2+2,:]
            self.m_JT[2*i + 1,:] = -(_rot * self.m_nhat[i,:].T).T*self.m_vc_jac[i*2:i*2+2,:]

        if self.simplify:
            self.m_JN = simplify(self.m_JN)
            self.m_JT = simplify(self.m_JT)

        self.m_JNS = self.m_JN[:,:len(self.sliders.v)]
        self.m_JNP = self.m_JN[:,len(self.sliders.v):]
        self.m_JTS = self.m_JT[:,:len(self.sliders.v)]
        self.m_JTP = self.m_JT[:,len(self.sliders.v):]

    def update_param(self):
        self.p_qs = Matrix(self.sliders.q.reshape(-1,3))
        self.p_qp = Matrix(self.pushers.q.reshape(-1,3))
        self.p_vs = Matrix(self.sliders.v.reshape(-1,3))
        self.p_vp = Matrix(self.pushers.v.reshape(-1,3))
        self.p_rs = Matrix(self.sliders.r.reshape(1,-1))
        self.p_rp = Matrix(self.pushers.r.reshape(1,-1))

    @property
    def q(self):
        return np.hstack([self.sliders.q, self.pushers.q])

    @property
    def qs(self):
        return self.sliders.q.reshape(-1)
    
    @property
    def qp(self):
        return self.pushers.q
    
    @property
    def v(self):
        return np.hstack([self.sliders.v, self.pushers.v])

    @property
    def phi(self):
        return np.array(self.m_phi.subs({self.sym_qs: self.p_qs,
                                         self.sym_qp: self.p_qp,
                                         self.sym_rs: self.p_rs,
                                         self.sym_rp: self.p_rp,
                                         })).astype(np.float64)

    @property
    def nhat(self):
        return np.array(self.m_nhat.subs({self.sym_qs: self.p_qs,
                                          self.sym_qp: self.p_qp,
                                          })).astype(np.float64)

    @property
    def vc(self):
        return np.array(self.m_vc.subs({self.sym_qs: self.p_qs,
                                        self.sym_qp: self.p_qp,
                                        self.sym_rs: self.p_rs,
                                        self.sym_rp: self.p_rp,
                                        self.sym_vs: self.p_vs,
                                        self.sym_vp: self.p_vp,
                                        })).astype(np.float64)
    @property
    def vc_jac(self):
        return np.array(self.m_vc_jac.subs({self.sym_qs: self.p_qs,
                                            self.sym_qp: self.p_qp,
                                            self.sym_rs: self.p_rs,
                                            self.sym_rp: self.p_rp,
                                            })).astype(np.float64)
    
    @property
    def JN(self):
        return np.array(self.m_JN.subs({self.sym_qs: self.p_qs,
                                        self.sym_qp: self.p_qp,
                                        })).astype(np.float64)

    @property
    def JNS(self):
        return np.array(self.m_JNS.subs({self.sym_qs: self.p_qs,
                                        self.sym_qp: self.p_qp,
                                        })).astype(np.float64)
    
    @property
    def JNP(self):
        return np.array(self.m_JNP.subs({self.sym_qs: self.p_qs,
                                        self.sym_qp: self.p_qp,
                                        })).astype(np.float64)
    @property
    def JT(self):
        return np.array(self.m_JT.subs({self.sym_qs: self.p_qs,
                                        self.sym_qp: self.p_qp,
                                        })).astype(np.float64)
    @property
    def JTS(self):
        return np.array(self.m_JTS.subs({self.sym_qs: self.p_qs,
                                        self.sym_qp: self.p_qp,
                                        self.sym_rs: self.p_rs,
                                        })).astype(np.float64)
    @property
    def JTP(self):
        return np.array(self.m_JTP.subs({self.sym_qs: self.p_qs,
                                        self.sym_qp: self.p_qp,
                                        })).astype(np.float64)
    
    @staticmethod
    def unit_vector(v):
        return v/sqrt(v.dot(v))
    
    @staticmethod
    def norm(v):
        return sqrt(v.dot(v))


    @staticmethod
    def combination(n, r):
        if n < r:
            return 0
        else:
            return int(math.factorial(n) / (math.factorial(n - r) * math.factorial(r)))