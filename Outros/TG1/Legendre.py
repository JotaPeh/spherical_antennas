# Main reference: https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special


# Legendre functions
# lpmv(m, v, x[, out]) -- Associated Legendre function of integer order and real degree.

# sph_harm(m, n, theta, phi[, out]) -- Compute spherical harmonics.

# The following functions do not accept NumPy arrays (they are not universal functions):

# clpmn(m, n, z[, type]) -- Associated Legendre function of the first kind for complex arguments.

# lpn(n, z) -- Legendre function of the first kind.

# lqn(n, z) -- Legendre function of the second kind.

# lpmn(m, n, z) -- Sequence of associated Legendre functions of the first kind.

# lqmn(m, n, z) -- Sequence of associated Legendre functions of the second kind.


# Hypergeometric functions
# hyp2f1(a, b, c, z[, out]) -- Gauss hypergeometric function 2F1(a, b; c; z)
# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp2f1.html#scipy.special.hyp2f1
# See also https://dlmf.nist.gov/14.3#i

# hyp1f1(a, b, x[, out]) -- Confluent hypergeometric function 1F1.

# hyperu(a, b, x[, out]) -- Confluent hypergeometric function U

# hyp0f1(v, z[, out]) -- Confluent hypergeometric limit function 0F1.


# Spherical Bessel functions
# spherical_jn(n, z[, derivative]) -- Spherical Bessel function of the first kind or its derivative.

# spherical_yn(n, z[, derivative]) -- Spherical Bessel function of the second kind or its derivative.

# spherical_in(n, z[, derivative]) -- Modified spherical Bessel function of the first kind or its derivative.

# spherical_kn(n, z[, derivative]) -- Modified spherical Bessel function of the second kind or its derivative.


# Riccati-Bessel functions
# The following functions do not accept NumPy arrays (they are not universal functions):

# riccati_jn(n, x) -- Compute Ricatti-Bessel function of the first kind and its derivative.

# riccati_yn(n, x) -- Compute Ricatti-Bessel function of the second kind and its derivative.

## Riccati
# def ricj(n, y):
#     return np.array([sp.riccati_jn(n, yi) for yi in y])

# def sjn(n, y):
#     return np.array([sp.spherical_jn(n, yi) for yi in y])

# def syn(n, y):
#     return np.array([sp.spherical_yn(n, yi) for yi in y])

# Zet = ricj(0,y)
# print(Zet)
# H = y*sjn(1,y)
# print(H)

# plt.plot(y, np.real(Zet), label='legendreQ')
# plt.plot(y, np.real(H), label='zero?')
# plt.title('Legendre Q Function')
# plt.grid(True)
# plt.show()

from scipy import special as sp
from scipy.optimize import root_scalar
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

phi1c = np.pi/3
phi2c = 2*np.pi/3

def legP(z, l, m):
    u = m * np.pi / (phi2c - phi1c)
    return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

def legQ(z, l, m):
    u = m * np.pi / (phi2c - phi1c)
    return np.where(np.abs(z) < 1, np.pi * (np.cos(u*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
    - sp.gamma(l+u+1) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / (sp.gamma(1+u) * sp.gamma(l-u+1))) / (2 * np.sin(u*np.pi)), 1)

    # np.sqrt(np.pi) * sp.gamma(l+u+1) * np.power(z**2-1,u/2) * sp.hyp2f1(l/2+u/2+1/2,l/2+u/2+1,l+3/2,1/z**2)  / (sp.gamma(l+3/2) * np.power(z,l+u+1) * 2**(l+1))
    # Implementar versões para u inteiro

y = np.linspace(-20, 20, 1000)

## Função esférica de Bessel de primeira espécie e ordem zero
n = 0
Y = sp.spherical_jn(n, y)
Z = np.sin(y)/y

plt.plot(y, Y, label='bessel')
plt.plot(y, Z, '--', label='sinc')
plt.xlabel('x')
plt.title('Função esférica de Bessel de ordem 0')
plt.grid(True)
plt.show()


## Legendre
x = np.linspace(-1, 1, 1000)[1:][:-1]

P=legP(x, 12, 0.75)
Q=legP(x, 5, 0.5)
# Coincide com LegendreP(12,3*2,x) e LegendreQ(5,3*1,x)

legP_wrapper = partial(legP, l=12, m=0.75)
legQ_wrapper = partial(legQ, l=5, m=0.5)
# print(root_scalar(legP_wrapper, x0=0).root)

roots = []
for i in range(len(x)-1):
    if legP_wrapper(x[i]) * legP_wrapper(x[i+1]) < 0:
        root = root_scalar(legP_wrapper, bracket=[x[i], x[i+1]], method='bisect')
        roots.append(root.root)
print("Raízes encontradas para P:", roots)

roots = []
for i in range(len(x)-1):
    if legQ_wrapper(x[i]) * legQ_wrapper(x[i+1]) < 0:
        root = root_scalar(legQ_wrapper, bracket=[x[i], x[i+1]], method='bisect')
        roots.append(root.root)
print("Raízes encontradas para Q:", roots)

plt.plot(x, P, label='legendreP')
plt.title('Legendre P Function')
plt.grid(True)
plt.show()

plt.plot(x, Q, label='legendreQ')
plt.title('Legendre Q Function')
plt.grid(True)
plt.show()