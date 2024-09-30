import numpy as np
from utils.lcp_solver import LCPSolver

class QuasiStateSim(object):
    '''
    Quasi-state simulator
    '''
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def run(self, u_input, qs, qp, phi, JNS, JNP, JTS, JTP):

        n_c   = phi.shape[0] # number of contact pairs
        n_s   = JNS.shape[1] # number of slider q
        n_p   = JNP.shape[1] # number of pusher q
        l = n_c * 3

        mu, A, B = self.sim_matrix(
            n_contac = n_c,
            n_slider = n_s,
            n_pusher = n_p,
            fascale  = 10, 
            fbscale  = 1,
            )

        E = np.repeat(np.eye(n_c), 2, axis=0)
        
        ZE = E * 0
        Z = np.zeros((n_c, n_c))

        M = np.zeros((4 * n_c, 4 * n_c))
        w = np.zeros(4 * n_c)

        JS = np.concatenate((JNS, JTS), axis = 0)
        JP = np.concatenate((JNP, JTP), axis = 0)

        M[:l,:l] = JS.dot(A.dot(JS.T))                  + JP.dot(B.dot(JP.T))
        M[:l,l:] = np.concatenate((Z, E), axis = 0)     + np.concatenate((Z, ZE), axis = 0)
        M[l:,:l] = np.concatenate((mu, -E.T), axis = 1) + np.concatenate((Z, ZE.T), axis = 1)
        M[l:,l:] = 2 * Z

        w[:l] = JP.dot(u_input)
        w[:n_c] += phi.reshape(-1)

        sol = LCPSolver(M = M,
                        q = w,
                        maxIter = self.n_steps
                        ).solve()

        if sol[0] is None:
            print('Solver failed: ', sol)
            print(phi)
            return qs, qp + u_input

        _qs = qs + A.dot(JS.T.dot(sol[0][:l]))
        # _qp = qp + B.dot(JP.T.dot(sol[0][:l])) + u_input
        _qp = qp + u_input

        return _qs, _qp
    
    def sim_matrix(self, n_contac, n_slider, n_pusher, fascale, fbscale):
        _mu = np.eye(n_contac)
        _A = np.eye(n_slider) * fascale
        _B = np.eye(n_pusher) * fbscale
        return _mu, _A, _B