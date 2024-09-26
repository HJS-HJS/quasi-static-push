from sympy import Matrix, MatrixSymbol, sqrt, zeros, ones, GramSchmidt
import numpy as np

def unit_vector(v):
    return v/norm(v)

def norm(v):
    return sqrt(v.dot(v))

m_test = zeros(1, 6)

qs = MatrixSymbol('qs_', 2, 3)
qp = MatrixSymbol('qp_', 2, 3)
rs = MatrixSymbol('rs_', 1, 2)
rp = MatrixSymbol('rp_', 1, 2)
vs = MatrixSymbol('vs_', 2, 3)
vp = MatrixSymbol('vp_', 2, 3)
sym_qs = Matrix(qs)
sym_qp = Matrix(qp)
sym_rs = Matrix(rs)
sym_rp = Matrix(rp)
sym_vs = Matrix(vs)
sym_vp = Matrix(vp)
sym_v = Matrix([sym_vs.col_join(sym_vp)[:]])
print('sym_qs\n\t', sym_qs)
print('sym_qp\n\t', sym_qp)
print('sym_rs\n\t', sym_rs)
print('sym_rp\n\t', sym_rp)
print('sym_vs\n\t', sym_vs)
print('sym_vp\n\t', sym_vp)
print('sym_v\n\t', sym_v)

m_phi = zeros(1, 4)
m_nhat = zeros(4, 2)
m_vc = zeros(1, 8)
_rot_arr = Matrix([[0, -1], [1, 0]])

i = 0
for i_s in range(2):
    for i_p in range(2):
        m_phi[i] = norm(sym_qp[i_p,0:2] - sym_qs[0,0:2]) - sym_rp[i_p] - sym_rs[i_s]
        m_nhat[i,:] = unit_vector(sym_qp[i_p,0:2] - sym_qs[0,0:2])
        i += 1
i = 0
for i_s in range(2):
    for i_p in range(2):
        point_vel = (sym_vs[0,2] * sym_rs[i_s] * _rot_arr * m_nhat[i,:].T).T + sym_vs[0,:2]
        m_vc[0,i*2:i*2+2] = sym_vp[i_p,0:2] - point_vel
        i += 1

# m_phi_jac = m_phi.row(0).jacobian(sym_qp.row(0))
m_vc_jac = m_vc.row(0).jacobian(sym_v.row(0))

print('m_phi')
print(m_phi)
print('m_nhat')
print(m_nhat)
print('vc')
print(m_vc)
print('vc_jaco')
print(m_vc_jac)

qs_l = Matrix(np.array([[0, 1, 2], [3, 4, 5]]))
qp_l = Matrix(2,3,[5, 4, 3, 2, 1, 0])

rs_l = Matrix(np.array([[0.5, 0.2]]))
rp_l = Matrix(np.array([[0.5, 0.2]]))

# ans = np.array(m_nhat.subs({qs: qs_l, qp: qp_l})).astype(np.float64)
# print('m_nhat')
# print(ans)


ans = np.array(m_phi.subs({qs: qs_l,
                           qp: qp_l,
                           rs: rs_l,
                           rp: rp_l,
                           })).astype(np.float64)
print('m_nhat')
print(ans)