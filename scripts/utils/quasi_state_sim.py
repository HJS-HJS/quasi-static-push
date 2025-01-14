import numpy as np
from utils.lcp_solver import LCPSolver

class QuasiStateSim(object):
    '''
    Quasi-state simulator
    '''
    def __init__(self, n_steps:int = 100):
        self.n_steps = n_steps

    def run(self, u_input, qs, qp, phi, JNS, JNP, JTS, JTP, mu, A, B, perfect_u_control:bool=True):

        phi, JNS, JNP, JTS, JTP, mu = self.clipping(phi, JNS, JNP, JTS, JTP, mu)

        if len(phi) == 0: return qs, qp + u_input, True

        n_c = phi.shape[0] # number of contact pairs
        l   = n_c * 3

        E = np.repeat(np.eye(n_c), 2, axis=0)
        
        ZE = E * 0
        Z = np.zeros((n_c, n_c))

        M = np.zeros((4 * n_c, 4 * n_c))
        w = np.zeros(4 * n_c)

        JS = np.concatenate((JNS, JTS), axis = 0)
        JP = np.concatenate((JNP, JTP), axis = 0)
        
        M[:l,:l] = JS.dot(A.dot(JS.T))                  
        M[:l,l:] = np.concatenate((Z, E), axis = 0)     + np.concatenate((Z, ZE), axis = 0)
        M[l:,:l] = np.concatenate((mu, -E.T), axis = 1) + np.concatenate((Z, ZE.T), axis = 1)
        M[l:,l:] = Z                                    + Z

        if not perfect_u_control: M[:l,:l] += JP.dot(B.dot(JP.T))

        w[:l] = JP.dot(u_input)
        w[:n_c] += phi.reshape(-1)

        sol = LCPSolver(M = M,
                        q = w,
                        maxIter = self.n_steps
                        ).solve()

        if sol[0] is None:
            # print('[0] Solver failed: ', sol)
            return qs, qp + u_input, True

        _d_qs = A.dot(JS.T.dot(sol[0][:l]))
        if np.max(_d_qs) > 0.01:
            # print('[1] Solver jump: ', sol)
            return qs, qp, False
        
        _qs = qs + _d_qs
        _qp = qp + u_input
        if not perfect_u_control: _qp += B.dot(JP.T.dot(sol[0][:l]))

        return _qs, _qp, True
    
    def clipping(self, phi, JNS, JNP, JTS, JTP, mu):
        _thres_idx = np.where(phi < 1e-3)
        _thres_idx_twice = np.repeat(_thres_idx,2) * 2

        return phi[_thres_idx], \
               JNS[_thres_idx], \
               JNP[_thres_idx], \
               JTS[_thres_idx_twice], \
               JTP[_thres_idx_twice], \
               mu[:len(_thres_idx[0]),:len(_thres_idx[0])]