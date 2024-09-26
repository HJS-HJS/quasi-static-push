from sympy import symbols, Matrix, sqrt, zeros, ones, GramSchmidt, sympify
import numpy as np

def unit_vector(v):
    return v/norm(v)

def norm(v):
    return sqrt(v.dot(v))

m_test = zeros(1, 6)

sym_qs = Matrix(2, 3, symbols(f"qs_0:{6}"))
sym_qp = Matrix(2, 3, symbols(f"qp_0:{6}"))
sym_rs = Matrix(symbols(f"rs_0:{2}"))
sym_rp = Matrix(symbols(f"rp_0:{2}"))
sym_vs = Matrix(2, 3, symbols(f"vs_0:{6}"))
sym_vp = Matrix(2, 3, symbols(f"vp_0:{6}"))
sym_v = Matrix([sym_vs.col_join(sym_vp)[:]])
print('sym_qs\n\t', sym_qs)
print('sym_qp\n\t', sym_qp)
print('sym_rs\n\t', sym_rs)
print('sym_rp\n\t', sym_rp)
print('sym_vs\n\t', sym_vs)
print('sym_vp\n\t', sym_vp)
print('sym_v\n\t', sym_v)
print('m_test\n\t', m_test)

# m_test[0,0:3] = ones(1,3)
# print('m_test\n\t', m_test)

# print(sym_qs - sym_vs)
# print('sym_qs.norm')
# print(sym_qs.norm)
# print('sym_qs**2')
# print(sym_qs[:2,:2]**2)
# print('sym_qs * sym_qs[0:3]')
# print(sym_qs)
# print(sym_qs[0,:])
# print(sym_qs * sym_qs[0,:].T)
# print((sym_qs * sym_qs[0,:].T)[0,:])
# print(sym_qs[:])
# jacobian = (sym_qs * sym_qs[0,:].T)[0,:].jacobian(sym_qs)
# print(jacobian)

# m_test = zeros(1, 6)
# for i in range(len(m_test)):
#     m_test[i] = sym_vs[i] * sym_vs[5 - i]

# print('m_test\n\t', m_test)
# print('m_test\n\t', m_test.row(0))
# print('m_test\n\t', sym_vs.row(0))
# print('m_test\n\t', m_test.row(0).jacobian(sym_vs.row(0)))

m_phi = zeros(1, 4)
m_nhat = zeros(4, 2)
m_vc = zeros(1, 8)
_rot_arr = Matrix([[0, -1], [1, 0]])
# test = Matrix([[2, 3]])
# print(_rot_arr)
# print(test.row(0))
# print(_rot_arr * test.row(0).T)

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

print('m_nhat')
print(m_nhat)
print(sympify(m_nhat))
# print('vc')
# print(m_vc)
# print('vc_jaco')
# print(m_vc_jac)

# XX, YY = symbols('XX, YY')
# x_set = {'XX': 5, "XY": 15}
# y_set = {'YY': 10, "YX": 15}
# xy_set = {'XX': 5, "XY": 15, 'YY': 10, "YX": 15}
# m_test = Matrix([[XX, YY], [XX * YY, XX + YY]])
# print(m_test)
# print(m_test.subs({'XX':5, 'YY': 10, 'ZZ':20}))
# print(m_test.subs(x_set))
# print(m_test.subs(y_set))
# print(m_test.subs(xy_set))
# print(m_test.subs(x_set, y_set))


# def value_hat():
#     m_nhat.subs({})

# ans = value_hat()
# print(ans)
