import numpy as np
from utils.lcp_solver import LCPSolver

class QuasiStateSim(object):
    '''
    Quasi-state simulator
    '''
    def __init__(self, dt, n_steps):
        self.dt = dt
        self.n_step = n_steps

    def run(self, u_input, qs, qp, phi, JNS, JNP, JTS, JTP):

        mu, A, B = self.sim_matrix(1, 0.1)

        m = JNS.shape[0]
        E = np.tile(np.eye(m), reps = [1, 2])
        E = np.array([
            E[0::m,:],
            E[1::m,:],
            ]).reshape(2*m, m)
        ZE = E * 0
        Z = np.zeros((m, m))

        M = np.zeros((3 * m + m, 3 * m + m))
        w = np.zeros(3 * m + m)

        JS = np.concatenate((JNS, JTS), axis = 0)
        JP = np.concatenate((JNP, JTP), axis = 0)

        M[:6,:6] = JS.dot(A.dot(JS.T))                  + JP.dot(B.dot(JP.T))
        M[:6,6:] = np.concatenate((Z, E), axis = 0)     + np.concatenate((Z, ZE), axis = 0)
        M[6:,:6] = np.concatenate((mu, -E.T), axis = 1) + np.concatenate((Z, ZE.T), axis = 1)
        M[6:,6:] = 2 * Z

        w[:6] = self.dt * JP.dot(u_input)
        w[:2] += phi.reshape(-1)

        sol = LCPSolver(M,w).solve()

        if sol[0] is None:
            print('Solver failed')
            return qs, qp + self.dt * u_input

        _qs = qs + A.dot(JS.T.dot(sol[0][:6]))
        _qp = qp + B.dot(JP.T.dot(sol[0][:6])) + self.dt * u_input
        # _qp = qp + self.dt * u_input

        return _qs, _qp
    
    def sim_matrix(self, fascale, fbscale):
        _A = np.eye(3) * fbscale
        _B = np.eye(3) * fbscale
        _mu = np.eye(2)
        return _mu, _A, _B