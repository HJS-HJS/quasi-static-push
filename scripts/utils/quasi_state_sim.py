import numpy as np
from utils.lcp_solver import LCPSolver
# import quantecon as qe

class QuasiStateSim(object):
    '''
    Quasi-state simulator
    '''
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def run(self, u_input, qs, qp, phi, JNS, JNP, JTS, JTP):
        if len(phi) is 0:
            return qs, qp + u_input
        n_c   = phi.shape[0] # number of contact pairs
        n_s   = JNS.shape[1] # number of slider q
        n_p   = JNP.shape[1] # number of pusher q
        l = n_c * 3

        mu, A, B = self.sim_matrix(
            n_contac = n_c,
            n_slider = n_s,
            n_pusher = n_p,
            # fmscale  = 0.8,
            # fascale  = 5000, 
            # fbscale  = 0.01,
            fmscale  = 0.8,
            fascale  = 5,
            fbscale  = 0.1,
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
        M[l:,l:] = Z                                    + Z

        w[:l] = JP.dot(u_input)
        w[:n_c] += phi.reshape(-1)

        sol = LCPSolver(M = M,
                        q = w,
                        maxIter = self.n_steps
                        ).solve()

        # sol =qe.optimize.lcp_lemke(M,w)
        # print(sol)
        if sol[0] is None:
            print('[0] Solver failed: ', sol)
            return qs, qp + u_input

        if np.max(A.dot(JS.T.dot(sol[0][:l]))) > 0.035:
            print('[1] Solver jump: ', sol)
            print(A.dot(JS.T.dot(sol[0][:l])))
            return qs, qp + u_input
        
        _qs = qs + A.dot(JS.T.dot(sol[0][:l]))
        _qp = qp + B.dot(JP.T.dot(sol[0][:l])) + u_input

        return _qs, _qp
    
    def sim_matrix(self, n_contac, n_slider, n_pusher, fmscale, fascale, fbscale):
        _mu = np.eye(n_contac) * fmscale
        _A  = np.eye(n_slider) * fascale
        _B  = np.eye(n_pusher) * fbscale
        _A[0,0] /= 10
        _A[1,1] /= 10
        _A[2,2] *= 1
        return _mu, _A, _B