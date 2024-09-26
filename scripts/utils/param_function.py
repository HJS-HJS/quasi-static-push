import numpy as np
from sympy import Matrix, MatrixSymbol, zeros, sqrt, simplify

class ParamFunction(object):
    '''
    Calculate objects param
    '''
    def __init__(self, sliders, pushers):
        self.sliders = sliders
        self.pushers = pushers

        self.initialize_param()

        self.show_symbols()

        _n_q    = len(self.q)
        _n_v    = len(self.v)
        # _n_phi  = len(self.phi)
        # _n_nhat = len(self.nhat)
        # self.JN = np.zeros((_n_phi, _n_q))
        # self.JT = np.zeros((2 * _n_phi, _n_v))
        # print('self.JN', self.JN.shape, self.JN)
        # print('self.JT', self.JT.shape, self.JT)

    def show_symbols(self):
        # print('self.m_qs\n', self.m_qs)
        # print('self.m_qp\n', self.m_qp)
        # print('self.m_rs\n', self.m_rs)
        # print('self.m_rp\n', self.m_rp)
        # print('self.m_vs\n', self.m_vs)
        # print('self.m_vp\n', self.m_vp)

        print('self.m_phi\n',       len(self.m_phi), self.m_phi)
        print('self.m_nhat\n',      len(self.m_nhat), self.m_nhat)
        print('self.m_vc\n',        len(self.m_vc), self.m_vc)
        print('self.m_v_jaco\n',    len(self.m_vc_jac), self.m_vc_jac)

        # print('self.JN', self.JN.shape, self.JN)
        # print('self.JT', self.JT.shape, self.JT)
        pass

    def initialize_param(self):
        self.sym_qs = MatrixSymbol('qs_', len(self.sliders), 3)
        self.sym_qp = MatrixSymbol('qp_', len(self.pushers), 3)
        self.sym_rs = MatrixSymbol('rs_', 1, len(self.sliders))
        self.sym_rp = MatrixSymbol('rp_', 1, len(self.pushers))
        self.sym_vs = MatrixSymbol('vs_', len(self.sliders), 3)
        self.sym_vp = MatrixSymbol('vp_', len(self.pushers), 3)
        self.m_qs = Matrix(self.sym_qs)
        self.m_qp = Matrix(self.sym_qp)
        self.m_rs = Matrix(self.sym_rs)
        self.m_rp = Matrix(self.sym_rp)
        self.m_vs = Matrix(self.sym_vs)
        self.m_vp = Matrix(self.sym_vp)
        self.m_v = Matrix([self.m_vs.col_join(self.m_vp)[:]])

        self.m_phi  = zeros(len(self.pushers) * len(self.sliders), 1)
        self.m_nhat = zeros(len(self.pushers) * len(self.sliders), 2)
        self.m_vc   = zeros(len(self.pushers) * len(self.sliders), 2)
        _rot_arr = Matrix([[0, -1], [1, 0]])

        i = 0
        for i_s in range(len(self.sliders)):
            for i_p in range(len(self.pushers)):
                self.m_phi[i] = ParamFunction.norm(self.m_qp[i_p,0:2] - self.m_qs[i_s,0:2]) - self.m_rp[i_p] - self.m_rs[i_s]
                self.m_nhat[i,:] = ParamFunction.unit_vector(self.m_qp[i_p,0:2] - self.m_qs[i_s,0:2])
                point_vel = (self.m_vs[i_s,2] * self.m_rs[i_s] * _rot_arr * self.m_nhat[i,:].T).T + self.m_vs[i_s,:2]
                self.m_vc[i,:] = self.m_vp[i_p,0:2] - point_vel
                i += 1
                
        self.m_vc_jac = self.m_vc.reshape(1,len(self.pushers) * len(self.sliders) * 2).jacobian(self.m_v)

        print(self.m_phi.shape)
        print(self.m_nhat.shape)
        print(self.m_vc.shape)
        print(self.m_vc_jac.shape)
        print('len_q\t',  len(self.q))
        print('len_v\t',  len(self.v))
        print('len_mv\t', len(self.m_v))

        self.m_phi      = simplify(self.m_phi)
        self.m_nhat     = simplify(self.m_nhat)
        self.m_vc       = simplify(self.m_vc)
        self.m_vc_jac   = simplify(self.m_vc_jac)

        self.JN = zeros(len(self.m_phi), len(self.q))
        self.JT = zeros(2 * len(self.m_phi), len(self.v))

        for i in range(len(self.m_phi)):
            print("\njaco")
            print(len(self.JN[i,:]))
            print(self.JN[i,:])
            print(len(self.m_nhat[i,:]))
            print(self.m_nhat[i,:])
            print(self.m_vc_jac[i*2:i*2+2,:].shape)
            print(len(self.m_vc_jac[i*2:i*2+2,:]))
            print(self.m_vc_jac[i*2:i*2+2,:])

            print('self.m_vc[i,:]')
            print(self.m_vc[i,:])
            print(self.m_vc[i,:].jacobian(self.m_v).shape)
            print(self.m_vc[i,:].jacobian(self.m_v))


        self.JN = simplify(self.JN)
        self.JT = simplify(self.JN)

        self.JNS = self.JN[:,:len(self.sliders.q)]
        self.JNP = self.JN[:,len(self.sliders.q):]
        self.JTS = self.JT[:,:len(self.sliders.v)]
        self.JTP = self.JT[:,len(self.sliders.v):]

    def update_param(self):
        self.qs = Matrix(self.sliders.q.reshape(-1,3))
        self.qp = Matrix(self.pushers.q_pusher.reshape(-1,3))
        self.vs = Matrix(self.sliders.v.reshape(-1,3))
        self.vp = Matrix(self.pushers.v_pusher.reshape(-1,3))
        self.rs = Matrix(self.sliders.r.reshape(1, -1))
        self.rp = Matrix(self.pushers.r.reshape(1, -1))

    def update_jacobian(self):
        # for idx in range(len(self.phi)):
        pass
            # self.JN[i:] = self.nhat[:i]@

    @property
    def q(self):
        return np.hstack([self.sliders.q, self.pushers.q])

    @property
    def v(self):
        return np.hstack([self.sliders.v, self.pushers.v])

    @property
    def phi(self):
        return np.array(self.m_phi.subs({self.sym_qs: self.qs,
                                         self.sym_qp: self.qp,
                                         self.sym_rs: self.rs,
                                         self.sym_rp: self.rp,
                                         })).astype(np.float64)

    @property
    def nhat(self):
        return np.array(self.m_nhat.subs({self.sym_qs: self.qs,
                                          self.sym_qp: self.qp,
                                          })).astype(np.float64)

    @property
    def vc(self):
        return np.array(self.m_vc.subs({self.sym_qs: self.qs,
                                          self.sym_qp: self.qp,
                                          self.sym_rs: self.rs,
                                          self.sym_rp: self.rp,
                                          self.sym_vs: self.vs,
                                          self.sym_vp: self.vp,
                                          })).astype(np.float64)
    @property
    def vc_jac(self):
        return np.array(self.m_vc_jac.subs({self.sym_qs: self.qs,
                                            self.sym_qp: self.qp,
                                            self.sym_rs: self.rs,
                                            self.sym_rp: self.rp,
                                            })).astype(np.float64)
    @staticmethod
    def unit_vector(v):
        return v/sqrt(v.dot(v))
    
    @staticmethod
    def norm(v):
        return sqrt(v.dot(v))
