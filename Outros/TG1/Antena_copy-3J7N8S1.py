from scipy import special as sp
from scipy import integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import warnings

show = 0

# Constantes gerais
dtr = np.pi/180         # Graus para radianos (rad/°)
e0 = 8.854e-12          # (F/m)
u0 = np.pi*4e-7         # (H/m)
c = 1/np.sqrt(e0*u0)    # Velocidade da luz no vácuo (m/s)
gamma = 0.5772156649015328606065120900824024310421 # Constante de Euler-Mascheroni
eps = 1e-5              # Limite para o erro numérico

# Geometria da cavidade
a = 100e-3              # Raio da esfera de terra (m)
h = 1.524e-3            # Espessura do substrato dielétrico (m)
a_bar = a + h/2         # Raio médio da esfera de terra (m)
b = a + h               # Raio exterior do dielétrico (m)
theta1c = 66.73 * dtr   # Ângulo de elevação 2 da cavidade (rad)
theta2c = 113.27 * dtr  # Ângulo de elevação 2 da cavidade (rad)
phi1c = 72.4 * dtr      # Ângulo de azimutal 2 da cavidade (rad)
phi2c = 107.6 * dtr     # Ângulo de azimutal 2 da cavidade (rad)

deltatheta1 = 0.1 * dtr # Largura angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = 0.1 * dtr # Largura angular 2 do campo de franjas e da abertura polar (rad)
theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2 # Ângulos polares físicos do patch (rad)
deltaPhi = 0.1 * dtr    # Largura angular do campo de franjas e da abertura azimutal (rad)
phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi               # Ângulos azimutais físicos do patch (rad)

# Campos distantes do modo TM01
def hankel_spher2(n, x, der = 0):
    # return sp.spherical_jn(n, x, der) - 1j * sp.spherical_yn(n, x, der)
    return sp.riccati_jn(n, x)[der]/x - 1j * sp.riccati_yn(n, x)[der]/x
def schelkunoff2(n, x, der = 1):
    return sp.riccati_jn(n, x)[der] - 1j * sp.riccati_yn(n, x)[der]

def Itheta(theta, L, M):
    return sp.lpmn(M, L, np.cos(theta))[0][M, L] * np.sin(theta)

def IDtheta(theta, L, M):
    return sp.lpmn(M, L, np.cos(theta))[1][M, L] * np.sin(theta)

def Eth_v_prot(f, ML, MM, theta, phi):
    Eth0_A, Eth0_C = 1, 1
    k = 2 * np.pi * f * u0 * e0

    print(ML)
    
    if MM > ML:
        return 0
    S_lm = (2*ML+1) * sp.gamma(1+ML-MM) / (2 * ML * (ML+1) * sp.gamma(1+ML+MM))

    IdP_1 = sp.lpmn(MM, ML, np.cos((theta1+theta1c)/2))[1][MM,ML] * np.sin((theta1+theta1c)/2) * deltatheta1
    IdP_2 = sp.lpmn(MM, ML, np.cos((theta2+theta2c)/2))[1][MM,ML] * np.sin((theta2+theta2c)/2) * deltatheta2
    IP_1 = sp.lpmn(MM, ML, np.cos((theta1+theta1c)/2))[0][MM,ML] * deltatheta1
    IP_2 = sp.lpmn(MM, ML, np.cos((theta2+theta2c)/2))[0][MM,ML] * deltatheta2
    
    # 1j**ML * np.exp(-1j * k * r) / r ignorado por normalização
    return (k ** 2 * b * sp.lpmn(MM, ML, np.cos(theta))[1][MM,ML] * (Eth0_A * IdP_1 + Eth0_C * IdP_2)/ (MM * schelkunoff2(ML, k * b)[ML]) \
            + 1j * MM * sp.lpmn(MM, ML, np.cos(theta))[0][MM,ML] * (Eth0_A * IP_1 + Eth0_C * IP_2) / (np.sin(theta) * hankel_spher2(ML, k * b)) ) * \
            2 * np.sin(MM*(phi2-phi1)/2) * np.cos(MM * ((phi1 + phi2)/2 + phi)) / (np.pi * S_lm)

def Eph_v_prot(f, ML, MM, theta, phi):
    Eth0_A, Eth0_C = 1, 1
    k = 2 * np.pi * f * u0 * e0

    if MM > ML:
        return 0
    S_lm = (2*ML+1) * sp.gamma(1+ML-MM) / (2 * ML * (ML+1) * sp.gamma(1+ML+MM))

    IdP_1 = sp.lpmn(MM, ML, np.cos((theta1+theta1c)/2))[1][MM,ML] * np.sin((theta1+theta1c)/2) * deltatheta1
    IdP_2 = sp.lpmn(MM, ML, np.cos((theta2+theta2c)/2))[1][MM,ML] * np.sin((theta2+theta2c)/2) * deltatheta2
    IP_1 = sp.lpmn(MM, ML, np.cos((theta1+theta1c)/2))[0][MM,ML] * deltatheta1
    IP_2 = sp.lpmn(MM, ML, np.cos((theta2+theta2c)/2))[0][MM,ML] * deltatheta2
    
    # -1j**ML * np.exp(-1j * k * r) / r ignorado por normalização
    return (k ** 2 * b * sp.lpmn(MM, ML, np.cos(theta))[0][MM,ML] * (Eth0_A * IdP_1 + Eth0_C * IdP_2)/ (np.sin(theta) * schelkunoff2(ML, k * b))[ML] \
            + 1j * sp.lpmn(MM, ML, np.cos(theta))[1][MM,ML] * (Eth0_A * IP_1 + Eth0_C * IP_2) / hankel_spher2(ML, k * b) ) * \
            2 * np.sin(MM*(phi2-phi1)/2) * np.sin(MM * ((phi1 + phi2)/2 + phi)) / (np.pi * S_lm)

def Eth_h_prot(f, ML, MM, theta, phi):
    Eph0 = 1
    k = 2 * np.pi * f * u0 * e0
    Dphic = phi1 - phi1c

    if MM > ML:
        return 0
    S_lm = (2*ML+1) * sp.gamma(1+ML-MM) / (2 * ML * (ML+1) * sp.gamma(1+ML+MM))

    I_th, error =  spi.quad(partial(Itheta, L = ML, M = MM), theta1 , theta2)
    I_dth, error =  spi.quad(partial(IDtheta, L = ML, M = MM), theta1 , theta2)
    
    # 1j**ML * np.exp(-1j * k * r) / r ignorado por normalização
    return (k ** 2 * b * I_th * sp.lpmn(MM, ML, np.cos(theta))[1][MM,ML] / schelkunoff2(ML, k * b)[ML] \
            + 1j * sp.lpmn(MM, ML, np.cos(theta))[0][MM,ML] * I_dth / (np.sin(theta) * hankel_spher2(ML, k * b)) ) * \
            4 * Eph0 * np.sin(MM*Dphic/2) * np.cos(MM*(phi2-phi1+Dphic)/2) * np.sin(MM * ((phi1 + phi2)/2 + phi)) / (np.pi * S_lm)

def Eph_h_prot(f, ML, MM, theta, phi):
    Eph0 = 1
    k = 2 * np.pi * f / c
    Dphic = phi1 - phi1c

    if MM > ML:
        return 0
    S_lm = (2*ML+1) * sp.gamma(1+ML-MM) / (2 * ML * (ML+1) * sp.gamma(1+ML+MM))

    I_th, error =  spi.quad(partial(Itheta, L = ML, M = MM), theta1 , theta2)
    I_dth, error =  spi.quad(partial(IDtheta, L = ML, M = MM), theta1 , theta2)
    
    # 1j**ML * np.exp(-1j * k * r) / r ignorado por normalização
    return (k ** 2 * MM * b * I_th * sp.lpmn(MM, ML, np.cos(theta))[0][MM,ML] / (np.sin(theta) * schelkunoff2(ML, k * b)[ML]) \
            + 1j * sp.lpmn(MM, ML, np.cos(theta))[1][MM,ML] * I_dth / (MM * hankel_spher2(ML, k * b)) ) * \
            4 * Eph0 * np.sin(MM*Dphic/2) * np.cos(MM*(phi2-phi1+Dphic)/2) * np.cos(MM * ((phi1 + phi2)/2 + phi)) / (np.pi * S_lm)

def E_v(f, ML, MM, theta, phi):
    # u = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eph_v_prot)(f, ML, MM, theta, phi))]))
    v = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eth_v_prot)(f, ML, MM, theta, phi))]))
    # tot = v**2 + u**2
    tot = v
    return 20*np.log10(tot/np.max(tot[~np.isnan(tot)]))

def E_h(f, ML, MM, theta, phi):
    u = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eph_h_prot)(f, ML, MM, theta, phi))]))
    # v = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eth_h_prot)(f, ML, MM, theta, phi))]))
    # tot = u**2 + v**2
    tot = u
    return 20*np.log10(tot/np.max(tot[~np.isnan(tot)]))

if show:
    inicio = time.time()
    angulos = np.arange(0,359,2) * dtr
    print(angulos)
    ML, MM, angs = np.meshgrid(np.arange(1,15), np.arange(1,15), angulos)
    print(np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eth_h_prot)(1.166e9, ML, MM, 90 * dtr, angs))])))
    fim = time.time()
    print("Tempo decorrido para o cálculo paralelo: ", fim - inicio, "segundos\n\n")

ML = 5
MM = 4
theta = 90 * dtr
phi = 90 * dtr

print()


Eth0_A, Eth0_C = 1, 1
k = 2 * np.pi * 1.166e9 / c

print(ML)

# if MM > ML:
#     return 0
S_lm = (2*ML+1) * sp.gamma(1+ML-MM) / (2 * ML * (ML+1) * sp.gamma(1+ML+MM))
print(S_lm)

IdP_1 = sp.lpmn(MM, ML, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2) * deltatheta1
print(IdP_1)
IdP_1 = IdP_1[MM,ML]
print(IdP_1)

IdP_2 = sp.lpmn(MM, ML, np.cos((theta2+theta2c)/2))[1][MM,ML] * np.sin((theta2+theta2c)/2) * deltatheta2
IP_1 = sp.lpmn(MM, ML, np.cos((theta1+theta1c)/2))[0][MM,ML] * deltatheta1
IP_2 = sp.lpmn(MM, ML, np.cos((theta2+theta2c)/2))[0][MM,ML] * deltatheta2

print(sp.spherical_jn(5, 5), sp.jv(5.5, 5) * np.sqrt(np.pi/(2*5)))
print('compare: ',sp.riccati_jn(5, 5)[0]/5)
print(sp.jvp(1.5, 5) * np.sqrt(np.pi*5/2)+sp.jv(1.5, 5) * np.sqrt(np.pi/(2*5*4)))
print(sp.riccati_jn(1, 5)[1])
print('real:',np.real(schelkunoff2(1,5)))

(k ** 2 * b * sp.lpmn(MM, ML, np.cos(theta))[1][MM,ML] * (Eth0_A * IdP_1 + Eth0_C * IdP_2)/ (MM * schelkunoff2(ML, k * b)[ML]) \
+ 1j * MM * sp.lpmn(MM, ML, np.cos(theta))[0][MM,ML] * (Eth0_A * IP_1 + Eth0_C * IP_2) / (np.sin(theta) * hankel_spher2(ML, k * b)) ) * \
2 * np.sin(MM*(phi2-phi1)/2) * np.cos(MM * ((phi1 + phi2)/2 + phi)) / (np.pi * S_lm)



angulos = np.arange(0,359,2) * dtr
ML, MM = np.meshgrid(np.arange(0,6), np.arange(0,5))

S_lm = (2*ML+1) * sp.gamma(1+ML-MM) / (2 * ML * (ML+1) * sp.gamma(1+ML+MM))
print(S_lm)


print('--------------------------------- \n\n\n\n\n')

def teste(theta, phi):
    L = M = 15 # M < L
    ML, MM = np.meshgrid(np.arange(0,L+1), np.arange(0,M+1))


    Eth0_A, Eth0_C = 1, 1
    k = 2 * np.pi * 1.166e9 / c
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_lm = (2*ML+1) * sp.gamma(1+ML-MM) / (2 * ML * (ML+1) * sp.gamma(1+ML+MM))

    IdP_1 = sp.lpmn(M, L, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2) * deltatheta1
    IdP_2 = sp.lpmn(M, L, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2) * deltatheta2
    IpP_1 = sp.lpmn(M, L, np.cos((theta1+theta1c)/2))[0] * deltatheta1
    IpP_2 = sp.lpmn(M, L, np.cos((theta2+theta2c)/2))[0] * deltatheta2

    # print(schelkunoff2(L, k * b))
    # print(np.tile(schelkunoff2(L, k * b),(M + 1, 1)))

    dH2_dr = np.tile(schelkunoff2(L, k * b),(M + 1, 1))
    H2 = np.tile(hankel_spher2(L, k * b),(M + 1, 1))

    # print(MM * dH_dr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc = ((1j ** ML) * (k ** 2 * b * sp.lpmn(M, L, np.cos(theta))[1] * (Eth0_A * IdP_1 + Eth0_C * IdP_2)/ (-np.sin(theta) * MM * dH2_dr) \
        + 1j * MM * sp.lpmn(M, L, np.cos(theta))[0] * (Eth0_A * IpP_1 + Eth0_C * IpP_2) / (np.sin(theta) * H2) ) * \
        2 * np.sin(MM*(phi2-phi1)/2) * np.cos(MM * ((phi1 + phi2)/2 + phi)) / (np.pi * S_lm))[1:]

    # print(np.sum(calc))

    tot = np.abs(np.sum(calc))
    # print(tot)
    return tot

# print(teste(90 * dtr, 90 * dtr))
# print(teste(40 * dtr, 90 * dtr))
eps = 0.01
tot = np.vectorize(teste)(90 * dtr +eps, np.arange(0,359,2) * dtr+eps)
# print(tot)
Ev = np.clip(20*np.log10(tot/np.max(tot[~np.isnan(tot)])), -30, 0)
print(Ev)

inicio = time.time()

fig, axs = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Plotar o primeiro gráfico
axs[0, 0].plot((angulos), Ev)
axs[0, 0].set_title('Amplitude do Campo Elétrico TM10 (dB)')
axs[0, 0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = 90$' + '°')
axs[0, 0].grid(True)
axs[0, 0].set_theta_zero_location('N')
axs[0, 0].set_theta_direction(-1)

tot1 = np.vectorize(teste)(np.arange(0,179,2) * dtr+eps, 90 * dtr+eps)
tot2 = np.vectorize(teste)(np.arange(0,179,2) * dtr+eps, -1 * 90 * dtr)
tots =  np.concatenate((tot1, tot2))
Etot = np.clip(20*np.log10(tots/np.max(tots[~np.isnan(tots)])), -30, 0)
axs[1, 0].plot(np.arange(0,359,2) * dtr, Etot, color='red')
axs[1, 0].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[1, 0].grid(True)
axs[1, 0].set_theta_zero_location('N')
axs[1, 0].set_theta_direction(-1)

fim = time.time()
print("Tempo decorrido para o cálculo dos 4 campos e gráficos: ", fim - inicio, "segundos\n\n")

plt.tight_layout()
plt.show()


print(sp.lpmn(4, 4, np.cos(52.45*dtr))[0] - sp.lpmn(4, 4, -np.cos(52.45*dtr))[0])