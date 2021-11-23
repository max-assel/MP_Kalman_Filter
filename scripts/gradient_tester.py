import numpy as np
from sympy import *

y1_, y2_, y3_, y4_ = symbols('y1 y2 y3 y4', real=True)
x1_, x2_, x3_, x4_ = symbols('x1 x2 x3 x4', real=True)
w1_, w2_, w3_, w4_ = symbols('w1 w2 w3 w4', real=True)
t_, t0_ = symbols('t t0')
place_holder = Symbol('x')
fx_jacobian = cos(place_holder) + 1
A_x_jacobian = cos(place_holder) + 1
fy_jacobian = cos(place_holder) + 1

x_k_kmin1 = [0, 0, 0.845, -0.453]

y_kmin1_kmin1 = [0, 0, 2.0, 1.0]

t0 = 0.0
t  = 1.0
y = [y1_, y2_, y3_, y4_]
x = [x1_, x2_, x3_, x4_]
f_x = Matrix([[y2_*sin(y3_) + y1_*cos(y3_)],
          [y2_*cos(y3_) - y1_*sin(y3_)],
          [sin(y3_)],
          [cos(y3_)]])

A_x = Matrix([[1, 0, 0, 0],
       [0, 1, 0, 0],
       [t_ - t0_, 0, 1, 0],
       [0, t_ - t0_, 0, 1]])

f_y = Matrix([(x1_*x4_ - x2_*x3_) / (x3_*x3_ + x4_*x4_),
              (x1_*x3_ + x2_*x4_) / (x3_*x3_ + x4_*x4_),
              (atan(x3_ / x4_)),
              1 / sqrt(x3_*x3_ + x4_*x4_)])

G1 = f_x.jacobian(y)
G1 = lambdify([y1_, y2_, y3_, y4_], G1, 'numpy')
G1_eval = G1(y_kmin1_kmin1[0], y_kmin1_kmin1[1], y_kmin1_kmin1[2], y_kmin1_kmin1[3])

G2 = A_x
G2 = lambdify([t_, t0_], G2, 'numpy')
G2_eval = G2(t, t0)

G3 = f_y.jacobian(x)
G3 = lambdify([x1_, x2_, x3_, x4_], G3, 'numpy')
G3_eval = G3(x_k_kmin1[0], x_k_kmin1[1], x_k_kmin1[2], x_k_kmin1[3])
print('G1_eval: ', G1_eval)
print('G2_eval: ', G2_eval)

print('G3_eval: ', G3_eval)

A_y = np.matmul(G3_eval, np.matmul(G2_eval, G1_eval))
print('A_y: ', A_y)
# G1 = jacobian of fx evaluated at y_kmin1_kmin1
# G2 = A_x
# G3 = jacobian of fy evaluated at x_k_kmin1


def obtain_symbolic_derivatives():
    global A_y

    S1 = y1_ + y4_*(w1_*cos(y3_) - w2_*sin(y3_))
    S2 = y2_ + y4_*(w1_*sin(y3_) + w2_*cos(y3_))
    S3 = (t_ + t0_)*y1_ - y4_*(w3_*cos(y3_) - w4_*sin(y3_))
    S4 = 1 + (t_ - t0_)*y2_ + y4_*(w3_*sin(y3_) + w4_*cos(y3_))
    f1 = (S1*S4 - S2*S3) / (S3*S3 + S4*S4)
    f2 = (S1*S3 + S2*S4) / (S3*S3 + S4*S4)
    f3 = y3_ + atan(S3/S4)
    f4 = y4_ / sqrt(S3*S3 + S4*S4)
    # print('diff S1: ', S1.diff(y3_subs))
    df1_dy1 = f1.diff(y1_)
    df1_dy2 = f1.diff(y2_)
    df1_dy3 = f1.diff(y3_)
    df1_dy4 = f1.diff(y4_)

    df2_dy1 = f2.diff(y1_)
    df2_dy2 = f2.diff(y2_)
    df2_dy3 = f2.diff(y3_)
    df2_dy4 = f2.diff(y4_)

    df3_dy1 = f3.diff(y1_)
    df3_dy2 = f3.diff(y2_)
    df3_dy3 = f3.diff(y3_)
    df3_dy4 = f3.diff(y4_)

    df4_dy1 = f4.diff(y1_)
    df4_dy2 = f4.diff(y2_)
    df4_dy3 = f4.diff(y3_)
    df4_dy4 = f4.diff(y4_)
    A_y = [[df1_dy1, df1_dy2, df1_dy3, df1_dy4],
           [df2_dy1, df2_dy2, df2_dy3, df2_dy4],
           [df3_dy1, df3_dy2, df3_dy3, df3_dy4],
           [df4_dy1, df4_dy2, df4_dy3, df4_dy4]]
    # print('A_y: ', A_y)
    A_y = lambdify([y1_, y2_, y3_, y4_,
                  t_, t0_,
                  w1_, w2_, w3_,w4_], A_y, 'numpy')
    # print('df1_dy2: ', df1_dy2)

