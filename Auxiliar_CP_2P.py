from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

filein = 'CP_2P_Param'
inicio_total = time.time()
show = 0

parametros = {line.split(' = ')[0]: float(line.split(' = ')[1]) for line in open('Resultados/Param/' + filein + '.txt')}
Dtheta, Dphi, thetap1, thetap2, phip1, phip2, re_ratioI, im_ratioI, flm_des = [parametros[key] for key in ['Dtheta', 'Dphi', 'thetap1', 'thetap2', 'phip1', 'phip2', 're_ratioI', 'im_ratioI', 'f']]
ratioI = re_ratioI + 1j*im_ratioI

# Constantes gerais
dtr = np.pi/180         # Graus para radianos (rad/°)
e0 = 8.854e-12          # (F/m)
u0 = np.pi*4e-7         # (H/m)
c = 1/np.sqrt(e0*u0)    # Velocidade da luz no vácuo (m/s)
gamma = 0.5772156649015328606065120900824024310421 # Constante de Euler-Mascheroni
eps = 1e-5              # Limite para o erro numérico

dtc = 0 * dtr
dpc = 0 * dtr
# Geometria da cavidade
a = 100e-3              # Raio da esfera de terra (m)
h = 1.524e-3            # Espessura do substrato dielétrico (m)
a_bar = a + h/2         # Raio médio da esfera de terra (m)
b = a + h               # Raio exterior do dielétrico (m)
theta1c = np.pi/2 - Dtheta/2 + dtc      # Ângulo de elevação 1 da cavidade (rad)
theta2c = np.pi/2 + Dtheta/2 + dtc      # Ângulo de elevação 2 da cavidade (rad)
phi1c = np.pi/2 - Dphi/2 + dpc          # Ângulo de azimutal 1 da cavidade (rad)
phi2c = np.pi/2 + Dphi/2 + dpc          # Ângulo de azimutal 2 da cavidade (rad)

Dtheta = theta2c - theta1c
Dphi = phi2c - phi1c
deltatheta1 = h/a       # Largura 32angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = h/a       # Largura angular 2 do campo de franjas e da abertura polar (rad)
theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2 # Ângulos polares físicos do patch (rad)
deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)
phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi             # Ângulos azimutais físicos do patch (rad)

# Coordenadas dos alimentadores coaxiais (rad)
thetap = [thetap1, thetap2]  
phip = [phip1, phip2]
# thetap = [90 * dtr, 94.66858482574794 * dtr]  
# phip = [94.73584985094497 * dtr, 90 * dtr]
probes = len(thetap)    # Número de alimentadores
df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
er = 2.55               # Permissividade relativa do substrato dielétrico
es = e0 * er            # Permissividade do substrato dielétrico (F/m)
Dphip = [np.exp(1.5)*df/(2*a*np.sin(t)) for t in thetap]     # Comprimento angular do modelo da ponta de prova (rad)

# Outras variáveis
tgdel = 0.0022           # Tangente de perdas
sigma = 5.8e50          # Condutividade elétrica dos condutores (S/m)
Z0 = 50                 # Impedância intrínseca (ohm)

path = 'HFSS/CP_2P/'

# Funções associadas de Legendre para |z| < 1
def legP(z, l, m):
    u = m * np.pi / (phi2c - phi1c)
    if u.is_integer():
        return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
    else:
        return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

def legQ(z, l, m):
    u = m * np.pi / (phi2c - phi1c)
    if u.is_integer():
        return np.where(np.abs(z) < 1, np.pi * (np.cos((u+l)*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
    - np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1+z)/2)  / sp.gamma(1-u)) / (2 * np.sin((u+l)*np.pi)), 1)
    else:
        return np.where(np.abs(z) < 1, np.pi * (np.cos(u*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
    - sp.gamma(l+u+1) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / (sp.gamma(1+u) * sp.gamma(l-u+1))) / (2 * np.sin(u*np.pi)), 1)

# Derivadas
def DP(theta, l, m):
    z = np.cos(theta)
    u = m * np.pi / (phi2c - phi1c)
    return l*legP(z, l, m)/np.tan(theta) - (l+u) * legP(z, l-1, m) / np.sin(theta)

def DQ(theta, l, m):
    z = np.cos(theta)
    u = m * np.pi / (phi2c - phi1c)
    return l*legQ(z, l, m)/np.tan(theta) - (l+u) * legQ(z, l-1, m) / np.sin(theta)

# Campos distantes dos modos TM01 e TM10
def hankel_spher2(n, x, der = 0):
    return sp.riccati_jn(n, x)[der]/x - 1j * sp.riccati_yn(n, x)[der]/x

def schelkunoff2(n, x, der = 1):
    return sp.riccati_jn(n, x)[der] - 1j * sp.riccati_yn(n, x)[der]

def Itheta(theta, L, M):
    if M > L:
        return 0
    return sp.lpmn(M, L, np.cos(theta))[0][M, L] * np.sin(theta)

def IDtheta(theta, L, M):
    if M > L:
        return 0
    return -sp.lpmn(M, L, np.cos(theta))[1][M, L] * np.sin(theta)**2

l = m = 35 # m <= l
Ml, Mm = np.meshgrid(np.arange(0,l+1), np.arange(0,m+1))
delm = np.ones(m+1)
delm[0] = 0.5

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    S_lm = (2 * Ml * (Ml+1) * sp.gamma(1+Ml+Mm)) / ((2*Ml+1) * sp.gamma(1+Ml-Mm))
    S_lm += (1-np.abs(np.sign(S_lm)))*1e-30
    I_th = np.array([[spi.quad(partial(Itheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
    I_dth = np.array([[spi.quad(partial(IDtheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])

# Início
def Equation(l, theta1f, theta2f, m):
    return np.where(np.abs(np.cos(theta1f)) < 1 and np.abs(np.cos(theta2f)) < 1, DP(theta1f,l,m)*DQ(theta2f,l,m)-DQ(theta1f,l,m)*DP(theta2f,l,m) , 1)

Lambda = np.linspace(-0.1, 32, 64) # Domínio de busca, o professor usou a janela de 0.1
rootsLambda = []

def root_find(n):
    roots = []
    Wrapper = partial(Equation, theta1f = theta1c, theta2f = theta2c, m = n)
    k = 0
    for i in range(len(Lambda)-1):
        if Wrapper(Lambda[i]) * Wrapper(Lambda[i+1]) < 0:
            root = root_scalar(Wrapper, bracket=[Lambda[i], Lambda[i+1]], method='bisect')
            roots.append(root.root)
            k += 1
    
    # Remoção de raízes inválidas
    k = 0
    for r in roots:
        # if round(r-roots[0], 6) % 1 == 0 and n != 0:
        if r < n * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
            k += 1
    if show:
        print("m =", n, ":", [round(r, 5) for r in roots[k:k+5]], ', mu =', n * np.pi / (phi2c - phi1c), "\n\n")
    rootsLambda.append(roots[k:k+5])

for n in range(0,5):
    root_find(n)

# Frequências de ressonância (GHz)
flm = []
if show:
    print('\nFrequências:')
for r in rootsLambda:
    flm.append([round(p,5) for p in 1e-9*np.sqrt([x*(x + 1) for x in r])/(2*np.pi*a_bar*np.sqrt(er*e0*u0))])
    if show:
        print([round(p,5) for p in 1e-9*np.sqrt([x*(x + 1) for x in r])/(2*np.pi*a_bar*np.sqrt(er*e0*u0))])
flm = np.transpose(flm) * 1e9                  # Hz
rootsLambda = np.transpose(rootsLambda)

klm = 2*np.pi*flm[0][1]*np.sqrt(u0*es)

def R(v,l,m):                                  # Função auxiliar para os campos elétricos
    L = rootsLambda[l][m]
    return (DP(theta1c,L,m)*legQ(v,L,m) - DQ(theta1c,L,m)*legP(v,L,m)) * np.sin(theta1c)

phi = np.linspace(phi1c, phi2c, 200)           # Domínio de phi (rad)
theta = np.linspace(theta1c, theta2c, 200)     # Domínio de theta (rad)
L, M = 0, 1                                    # Modo em teste

# Impedância de entrada - Circuito RLC:
def R2(v, l, m):                               # Quadrado da função auxiliar para os campos elétricos
    return R(v, l, m)**2

def tgefTot(klm, L, M):
    # Qdie = 2 * np.pi * f * es / sigma_die # + Perdas de irradiação = 1/tgdel
    Rs = np.sqrt(klm * np.sqrt(u0/es) / (2 * sigma))
    Qc = klm * np.sqrt(u0/es) * h / (2 * Rs)
    Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)
    
    if M == 0:
        K0 = klm / np.sqrt(er)

        I10 = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]

        IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
        IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
        IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
        IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2
        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S10 = (((phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) ** 2 / (S_lm)) * (np.abs(b*(IdP_1+IdP_2)/dH2_dr)**2 + Mm**2 * np.abs((IpP_1+IpP_2)/(K0*H2))**2)
        S10 = np.sum(np.dot(delm, S10))

        Q10 = (np.pi / 3) * klm * np.sqrt(er) * (b**3 - a**3) * I10 * Dphi / (np.abs(R(np.cos(theta1c),L,M))**2 * S10)
        return tgdel + 1/Qc + 1/Q10
    elif M == 1:
        K0 = klm / np.sqrt(er)

        I01 = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]

        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S01 = ((deltaPhi * np.sinc(Mm*deltaPhi/(2*np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2)) ** 2 / (S_lm)) * (Mm**2 * np.abs(b*I_th/dH2_dr)**2 + np.abs(I_dth/(K0*H2))**2)
        S01 = np.sum(np.dot(delm, S01))

        Q01 = (np.pi / 24) * klm * np.sqrt(er) * (b**3 - a**3) * I01 * Dphi / (np.abs(R(np.cos((theta1c+theta2c)/2),L,M))**2 * S01)
        return tgdel + 1/Qc + 1/Q01
    else:
        print('Erro ao calcular a tangente efeitiva de perdas!')

def RLC(f, klm, L, M, p1, p2):
    U = M * np.pi / (phi2c - phi1c)
    Ilm = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]

    # Auxiliar alpha
    alpha = h * (R(np.cos(thetap[p1-1]), L, M) * np.cos(U * (phip[p1-1]-phi1c)) * np.sinc(U * Dphip[p1-1] / (2*np.pi))) * \
        (R(np.cos(thetap[p2-1]), L, M) * np.cos(U * (phip[p2-1]-phi1c)) * np.sinc(U * Dphip[p2-1] / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
    if M != 0:
        alpha = 2 * alpha
    
    # Componentes RLC
    RLM = alpha/(2 * np.pi * f * tgefTot(klm, L, M))
    CLM = 1/alpha
    LLM = alpha/(2 * np.pi * flm[L][M])**2
    return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

def ZCP(f, p1, p2):
    k = (2 * np.pi * f) * np.sqrt(es * u0) 
    eta = np.sqrt(u0 / es)
    Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
    if p1 != p2:
        return RLC(f, klm, 0, 1, p1, p2) + RLC(f, klm, 1, 0, p1, p2)
    return RLC(f, klm, 0, 1, p1, p2) + RLC(f, klm, 1, 0, p1, p2) + 1j*Xp

freqs01 = np.linspace(1.45, 1.75, 641) * 1e9
freqs10 = np.linspace(1.45, 1.75, 641) * 1e9

# Razão Axial
tgef10 = tgefTot(klm, 1, 0)
tgef01 = tgefTot(klm, 0, 1)

def S(Kdes, Dtheta, Dphi, theta, phi):
    theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
    phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
    theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
    phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
    Dphic = phi1 - phi1c
    Kdes = Kdes/np.sqrt(er)

    IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
    IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
    IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
    IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2
    dH2_dr = np.tile(schelkunoff2(l, Kdes * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, Kdes * b),(m + 1, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sthv = ((1j ** Ml) * (b * sp.lpmn(m, l, np.cos(theta))[1] * (IdP_1 + IdP_2) * (-np.sin(theta))/ (dH2_dr) \
            + 1j * Mm**2 * sp.lpmn(m, l, np.cos(theta))[0] * (IpP_1 + IpP_2) / (Kdes * np.sin(theta) * H2) ) * \
            (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) * np.cos(Mm * ((phi1+phi2)/2-phi)) / (np.pi * S_lm)
        Sthv = np.dot(delm, Sthv)
        Sphh = ((1j ** Ml) * (Mm**2 * b * I_th * sp.lpmn(m, l, np.cos(theta))[0] / (dH2_dr * np.sin(theta)) \
            + 1j * sp.lpmn(m, l, np.cos(theta))[1] * -np.sin(theta) * I_dth / (Kdes * H2) ) * \
            2 * Dphic * np.sinc(Mm*Dphic/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+Dphic)/2)) * np.cos(Mm * ((phi1+phi2)/2-phi)) / (np.pi * S_lm)
        Sphh = np.dot(delm, Sphh)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sr = np.sum(Sthv)/np.sum(Sphh)
    return Sr

def RA(klm, freq, tgef01, tgef10, Dtheta, Dphi, thetap2, phip1, ratioI, theta, phi, field = 0):
    L01, M01 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, np.pi/Dphi
    L10, M10 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, 0
    theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
    phi1c = np.pi/2 - Dphi/2
    Dphip = np.exp(1.5)*df/(2*a*np.sin(np.pi/2))
    Kef01 = 2*np.pi*freq*np.sqrt(u0*es*(1-1j*tgef01))
    Kef10 = 2*np.pi*freq*np.sqrt(u0*es*(1-1j*tgef10))
    Kdes = 2*np.pi*freq*np.sqrt(u0*es)

    I01 = spi.quad(partial(R2, l = 0, m = 1),np.cos(theta2c),np.cos(theta1c))[0]
    I10 = spi.quad(partial(R2, l = 1, m = 0),np.cos(theta2c),np.cos(theta1c))[0]

    if field == 0:
        V = (I01 * R(np.cos(theta1c),1,0))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),0,1)) * (Kef01**2 - klm**2)/(Kef10**2 - klm**2) * S(Kdes, Dtheta, Dphi, theta, phi)
    else:
        V = (I01 * R(np.cos(theta1c),1,0))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),0,1)) * (Kef01**2 - klm**2)/(Kef10**2 - klm**2)

    return ratioI * (R(np.cos(thetap2),1,0) / (R(0,0,1) * np.cos(M01*(phip1 - phi1c)) * np.sinc(M01*Dphip/(2*np.pi)))) * V

# Campos distantes
eps = 1e-30

def Eth_v_prot(theta, phi):
    k = 2 * np.pi * flm_des / c
    Eth0_A = Eth0_C = 1
    
    IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
    IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
    IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
    IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2

    dH2_dr = np.tile(schelkunoff2(l, k * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, k * b),(m + 1, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # np.exp(-1j * k * r) / r ignorado por normalização
        Ethv = ((1j ** Ml) * (b * sp.lpmn(m, l, np.cos(theta))[1] * (Eth0_A * IdP_1 + Eth0_C * IdP_2) * (-np.sin(theta))/ (dH2_dr) \
            + 1j * Mm**2 * sp.lpmn(m, l, np.cos(theta))[0] * (Eth0_A * IpP_1 + Eth0_C * IpP_2) / (k * np.sin(theta) * H2) ) * \
            (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) * np.cos(Mm * ((phi1 + phi2)/2 - phi)) / (np.pi * S_lm))
        # if phi == theta and show:
        #     print(Ethv)
            # print(np.round(IpP_1-IpP_2,10))
        return np.sum(np.dot(delm, Ethv)) # V/m
        
def Eph_v_prot(theta, phi):
    k = 2 * np.pi * flm_des / c
    Eth0_A = Eth0_C = 1

    IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
    IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
    IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
    IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2

    dH2_dr = np.tile(schelkunoff2(l, k * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, k * b),(m + 1, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # np.exp(-1j * k * r) / r ignorado por normalização
        Ephv = ((1j ** Ml) * (b * sp.lpmn(m, l, np.cos(theta))[0] * (Eth0_A * IdP_1 + Eth0_C * IdP_2)/ (np.sin(theta) * dH2_dr) \
            + 1j * sp.lpmn(m, l, np.cos(theta))[1] * (Eth0_A * IpP_1 + Eth0_C * IpP_2) * (-np.sin(theta)) / (k * H2) ) * \
            2 * np.sin(Mm * (phi2-phi1)/2) * np.sin(Mm * ((phi1 + phi2)/2 - phi)) / (np.pi * S_lm))
        return np.sum(np.dot(delm, Ephv)) # V/m

def Eth_h_prot(theta, phi):
    k = 2 * np.pi * flm_des / c
    Dphic = phi1 - phi1c
    Eph0 = 1

    dH2_dr = np.tile(schelkunoff2(l, k * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, k * b),(m + 1, 1))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # np.exp(-1j * k * r) / r ignorado por normalização
        Ethh = ((-1j ** Ml) * (b * I_th * sp.lpmn(m, l, np.cos(theta))[1] * (-np.sin(theta)) / dH2_dr \
            + 1j * sp.lpmn(m, l, np.cos(theta))[0] * I_dth / (k * H2 * np.sin(theta)) ) * \
            4 * Eph0 * np.sin(Mm*Dphic/2) * np.cos(Mm*(phi2-phi1+Dphic)/2) * np.sin(Mm * ((phi1 + phi2)/2 - phi)) / (np.pi * S_lm))
        return np.sum(np.dot(delm, Ethh))

def Eph_h_prot(theta, phi):
    k = 2 * np.pi * flm_des / c
    Dphic = phi1 - phi1c
    Eph0 = 1

    dH2_dr = np.tile(schelkunoff2(l, k * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, k * b),(m + 1, 1))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # np.exp(-1j * k * r) / r ignorado por normalização
        Ephh = ((1j ** Ml) * (Mm**2 * b * I_th * sp.lpmn(m, l, np.cos(theta))[0] / (np.sin(theta) * dH2_dr) \
            + 1j * sp.lpmn(m, l, np.cos(theta))[1] * I_dth * (-np.sin(theta)) / (k *H2) ) * \
            2 * Eph0 * Dphic * np.sinc(Mm*Dphic/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+Dphic)/2) * np.cos(Mm * ((phi1 + phi2)/2 - phi)) / (np.pi * S_lm))
        return np.sum(np.dot(delm, Ephh))

def E_v(theta, phi):
    if isinstance(phi, np.ndarray):
        u = np.abs(np.vectorize(Eth_v_prot)(theta, phi))
        v = np.abs(np.vectorize(Eph_v_prot)(theta, phi))
    elif isinstance(theta, np.ndarray):
        u = np.abs(np.concatenate((np.vectorize(Eth_v_prot)(theta[:len(theta)//2], phi), np.vectorize(Eth_v_prot)(theta[:len(theta)//2], -phi)[::-1])))
        v = np.abs(np.concatenate((np.vectorize(Eph_v_prot)(theta[:len(theta)//2], phi), np.vectorize(Eph_v_prot)(theta[:len(theta)//2], -phi)[::-1])))
    tot = v**2 + u**2
    return np.clip(10*np.log10(tot/np.max(tot[~np.isnan(tot)])), -30, 0)

def E_h(theta, phi):
    if isinstance(phi, np.ndarray):
        u = np.abs(np.vectorize(Eth_h_prot)(theta, phi))
        v = np.abs(np.vectorize(Eph_h_prot)(theta, phi))
    elif isinstance(theta, np.ndarray):
        u = np.abs(np.concatenate((np.vectorize(Eth_h_prot)(theta[:len(theta)//2], phi), np.vectorize(Eth_h_prot)(theta[:len(theta)//2], -phi)[::-1])))
        v = np.abs(np.concatenate((np.vectorize(Eph_h_prot)(theta[:len(theta)//2], phi), np.vectorize(Eph_h_prot)(theta[:len(theta)//2], -phi)[::-1])))
    tot = u**2 + v**2
    return np.clip(10*np.log10(tot/np.max(tot[~np.isnan(tot)])), -30, 0)

def E_HCP(theta, phi, sentido):
    if isinstance(phi, np.ndarray):
        kRA = np.vectorize(RA)(klm, flm_des, tgef01, tgef10, Dtheta, Dphi, thetap[1], phip[0], ratioI, theta, phi)
        Eph = np.abs(np.vectorize(Eph_h_prot)(theta, phi))
    if isinstance(theta, np.ndarray):
        kRA = np.concatenate((np.vectorize(RA)(klm, flm_des, tgef01, tgef10, Dtheta, Dphi, thetap[1], phip[0], ratioI, theta[:len(theta)//2], phi),np.vectorize(RA)(klm, flm_des, tgef01, tgef10, Dtheta, Dphi, thetap[1], phip[0], ratioI, theta[:len(theta)//2], -phi)[::-1]))
        Eph = np.abs(np.concatenate((np.vectorize(Eph_h_prot)(theta[:len(theta)//2], phi), np.vectorize(Eph_h_prot)(theta[:len(theta)//2], -phi)[::-1])))
    if sentido == 'R':
        tot = np.abs(Eph*(kRA+1j))
    if sentido == 'L':
        tot = np.abs(Eph*(kRA-1j))
    return np.clip(20*np.log10(tot/np.max(tot[~np.isnan(tot)])), -30, 0)

#####################################

def Gain():
    Kef01 = (2 * np.pi * flm_des) * np.sqrt(es * u0 * (1-1j*tgefTot(klm, 0, 1)))
    Kef10 = (2 * np.pi * flm_des) * np.sqrt(es * u0 * (1-1j*tgefTot(klm, 1, 0)))
    k01 = 2*np.pi*flm[0][1]*np.sqrt(u0*es)
    k10 = 2*np.pi*flm[1][0]*np.sqrt(u0*es)
    K0 = klm / np.sqrt(er)

    eta0 = np.sqrt(u0 / e0)

    M01 = np.pi/Dphi

    I10 = spi.quad(partial(R2, l = 1, m = 0),np.cos(theta2c),np.cos(theta1c))[0]
    I01 = spi.quad(partial(R2, l = 0, m = 1),np.cos(theta2c),np.cos(theta1c))[0]

    IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
    IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
    IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
    IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2
    dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

    S10 = (((phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) ** 2 / (S_lm)) * (np.abs(b*(IdP_1+IdP_2)/dH2_dr)**2 + Mm**2 * np.abs((IpP_1+IpP_2)/(K0*H2))**2)
    Ckth = np.sum(np.dot(delm, S10)) / (2*np.pi*eta0)

    S01 = ((deltaPhi * np.sinc(Mm*deltaPhi/(2*np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2)) ** 2 / (S_lm)) * (Mm**2 * np.abs(b*I_th/dH2_dr)**2 + np.abs(I_dth/(K0*H2))**2)
    Ckph = 2*np.sum(np.dot(delm, S01)) / (np.pi*eta0)

    ki = Ckth * (R(np.cos(theta1c),1,0) * R(np.cos(thetap[1]),1,0) * np.abs(ratioI) / (np.abs(Kef10**2-k10**2)*I10))**2 \
        + Ckph * (2 * R(np.cos((theta1c+theta2c)/2),0,1) * R(0,0,1) * np.cos(M01*(phip[0] - phi1c)) * np.sinc(M01*Dphip[0]/(2*np.pi)) / (np.abs(Kef01**2-k01**2)*I01))**2

    ki *= (2 * np.pi * flm_des * u0 / (Dphi * a_bar**2))**2 / np.real((ZCP(flm_des, 1, 1) * np.abs(ratioI)**2 + ZCP(flm_des, 2, 2))/2)

    Ceth2 = (2 * np.pi * flm_des * u0 / (Dphi * a_bar**2))**2 * (R(np.cos(theta1c),1,0) * R(np.cos(thetap[1]),1,0) * np.abs(ratioI) / (np.abs(Kef10**2-k10**2)*I10))**2
    Ceph2 = (2 * np.pi * flm_des * u0 / (Dphi * a_bar**2))**2 * (2 * R(np.cos((theta1c+theta2c)/2),0,1) * R(np.cos(thetap[0]),0,1) * np.cos(M01*(phip[0] - phi1c)) * np.sinc(M01*Dphip[0]/(2*np.pi)) / (np.abs(Kef01**2-k01**2)*I01))**2

    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Sthv = ((1j ** Ml) * (- b * sp.lpmn(m, l, 0)[1] * (IdP_1 + IdP_2) / (dH2_dr) \
                + 1j * Mm**2 * sp.lpmn(m, l, 0)[0] * (IpP_1 + IpP_2) / (K0 * H2) ) * \
                (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) / S_lm)
            Sthv = np.sum(np.dot(delm, Sthv)) / np.pi
        
            Sphh = ((1j ** Ml) * (Mm**2 * b * I_th * sp.lpmn(m, l, 0)[0] / dH2_dr \
                + 1j * sp.lpmn(m, l, 0)[1] * -I_dth / (K0 * H2) ) * \
                2 * deltaPhi * np.sinc(Mm*deltaPhi/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2) / S_lm)
            Sphh = np.sum(np.dot(delm, Sphh)) / np.pi

    D = 2*np.pi/eta0 * (Ceth2 * np.abs(Sthv)**2 + Ceph2 * np.abs(Sphh)**2) / (Ceth2 * Ckth + Ceph2 * Ckph)

    G = ki * D
    
    if 1:
        print('ki = ', ki)
        print('D = ', D, ', dB: ', 10*np.log10(D))
        print('G = ', G, ', dB: ', 10*np.log10(G), '\n\n')
    
    # exit()
    
    # parametros = f"Modelo:\nDiretividade (dB) = {10*np.log10(D)}\nEficiencia (%) = {100*ki[0]}\nGanho (dBi) = {10*np.log10(G)}\n\n"
    # if M == 0:
    #     parametros += f"Simulacao:\nDiretividade (dB) = {10*np.log10(10**(GainLin/10)/0.864511)}\nEficiencia (%) = {86.4511}\nGanho (dBi) = {GainLin}" 
    # elif M == 1:
    #     parametros += f"Simulacao:\nDiretividade (dB) = {10*np.log10(10**(GainLin/10)/0.855087)}\nEficiencia (%) = {85.5087}\nGanho (dBi) = {GainLin}"
    # out_dir = 'Resultados'
    # out_file = os.path.join(out_dir, 'LP'+str(1-M)+str(M)+'_Ganho.txt')
    # os.makedirs(out_dir, exist_ok=True)
    # with open(out_file, 'w') as f:
    #     f.write(parametros)

Gain()

#####################################

print('Opção 1:\n') # Ephi e Etheta
def Alpha(ang):
    Ev =  E_v(np.array([ang+eps]+[1]*4), (phi1c + phi2c) / 2 + eps)[0]+3
    return Ev
root = root_scalar(Alpha, bracket=[2*dtr, np.pi/2-2*dtr], method='bisect')
alphaP = np.pi - 2*np.array(root.root)

# print(E_v(np.array([np.array(root.root)+eps]+[1]*4), (phi1c + phi2c) / 2 + eps)[0])

def Alpha(ang):
    Ev =  E_v((theta1c + theta2c) / 2 + eps, np.array([ang+eps]+[1]*4))[0]+3
    return Ev
root = root_scalar(Alpha, bracket=[2*dtr, np.pi/2-2*dtr], method='bisect')
alphaT = np.pi - 2*np.array(root.root)

Dv = 4*np.pi/(alphaT*alphaP)
print(alphaT/dtr, alphaP/dtr)

def Alpha(ang):
    Eh =  E_h(np.array([ang+eps]+[1]*4), (phi1c + phi2c) / 2 + eps)[0]+3
    return Eh
root = root_scalar(Alpha, bracket=[2*dtr, np.pi/2-2*dtr], method='bisect')
alphaP = np.pi - 2*np.array(root.root)

def Alpha(ang):
    Eh =  E_h((theta1c + theta2c) / 2 + eps, np.array([ang+eps]+[1]*4))[0]+3
    return Eh
root = root_scalar(Alpha, bracket=[2*dtr, np.pi/2-2*dtr], method='bisect')
alphaT = np.pi - 2*np.array(root.root)

Dh = 4*np.pi/(alphaT*alphaP)
print(alphaT/dtr, alphaP/dtr)

print('\n',Dv,Dh, 10*np.log10(Dv))
D_3dB = 10*np.log10(Dv+Dh)
print('Diretividade (3dB):', D_3dB)
#####################################

print('\n\nOpção 2:\n') # Erhcp

dataTh = pd.read_csv(path+'Gain Plot HCP theta.csv')
gainTh_T = dataTh['dB(GainRHCP) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
G_RHCP = np.max(gainTh_T)
dataPh = pd.read_csv(path+'Gain Plot HCP phi.csv')
gainPh_P = dataPh['dB(GainLHCP) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']
gainPh_P -= G_RHCP#np.max(gainPh_P)
gTest = np.max(gainPh_P)

angulos = np.arange(85,95,1) * dtr + eps

u =  E_HCP((theta1c + theta2c) / 2 + eps, angulos, 'L')+gTest
v = E_HCP((theta1c + theta2c) / 2 + eps, angulos, 'R')
Emax = np.max(10*np.log10(10**(u/10)+10**(v/10)))
# print(10*np.log10(10**(u/10)+10**(v/10)))
# print(Emax)

# print('here')
def Alpha(ang):
    u =  E_HCP((theta1c + theta2c) / 2 + eps, np.array([ang+eps]+[1]*4), 'L')[0]+gTest
    v = E_HCP((theta1c + theta2c) / 2 + eps, np.array([ang+eps]+[1]*4), 'R')[0]
    Eh = 10*np.log10(10**(u/10)+10**(v/10))
    # print(Eh-Emax+3)
    # print(v)
    return Eh-Emax+3
    return v+3
root = root_scalar(Alpha, bracket=[2*dtr, np.pi/2-2*dtr], method='bisect')
alphaP = np.pi - 2*np.array(root.root)

def Alpha(ang):
    u =  E_HCP(np.array([ang+eps]+[1]*4), (phi1c + phi2c) / 2 + eps, 'L')[0]+gTest
    v = E_HCP(np.array([ang+eps]+[1]*4), (phi1c + phi2c) / 2 + eps, 'R')[0]
    Eh = 10*np.log10(10**(u/10)+10**(v/10))
    # print(v)
    return Eh-Emax+3
    return v+3
root = root_scalar(Alpha, bracket=[2*dtr, np.pi/2-2*dtr], method='bisect')
alphaT = np.pi - 2*np.array(root.root)

Dh = 4*np.pi/(alphaT*alphaP)
print(alphaT/dtr, alphaP/dtr)

# print('\n\n',Dv,Dh, 10*np.log10(Dv))
D_3dB = 10*np.log10(Dh)
print('\nDiretividade (3dB):', D_3dB, '\n\n')

######################################
output_folder = 'Resultados/Analise_CP2_Hib'
os.makedirs(output_folder, exist_ok=True)
figures = []

path = 'HFSS/CP2_Hib/'

# Axial Ratio
data_hfss = pd.read_csv(path+'Axial Ratio Plot f.csv')
freqs_hfss = data_hfss['F [GHz]'] # GHz
ra_hfss = data_hfss['dB(AxialRatioValue) [] - Phi=\'90.0000000000002deg\' Theta=\'90.0000000000002deg\'']

fig = plt.figure()
plt.plot(freqs_hfss, ra_hfss, label='RA simulado')
plt.axvline(x=flm_des / 1e9, color='r', linestyle='--')
# plt.axhline(y=3, color='g', linestyle='--')
plt.xlabel('Frequência (GHz)')
plt.ylabel('Razão Axial (dB)')
plt.title('Razão Axial')
plt.grid(True)
plt.legend()
figures.append(fig)

# Gamma_in = s11
data_hfss = pd.read_csv(path+'S Parameter CP2.csv')
freqs_hfss = data_hfss['F [GHz]'] # GHz
dB_s11 = data_hfss['dB(S(Port1,Port1)) []'] # dB

fig = plt.figure()
plt.plot(freqs_hfss, dB_s11, label=r'$|s_{11}|$ simulado')
# plt.axhline(y=-10, color='b', linestyle='--')
plt.axvline(x=flm_des / 1e9, color='r', linestyle='--')
plt.xlabel('Frequência (GHz)')
plt.ylabel(r'$|s_{11}|$' + ' (dB)')
plt.title('Parâmetro '+r'$|s_{11}|$')
plt.legend(loc='upper left')
plt.grid(True)
figures.append(fig)

# Ganho RHCP
angulos = np.arange(0,360,1) * dtr + eps
angles = np.arange(0, 360, 30)

fig, axs = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={'projection': 'polar'})

dataTh = pd.read_csv(path+'Gain Plot HCP theta.csv')
angTh = dataTh['Phi [deg]'] * dtr + eps
gainTh_T = dataTh['dB(GainRHCP) [] - F=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
G_RHCP = np.max(gainTh_T)
gainTh_T -= G_RHCP#np.max(gainTh_T)
gainTh_P = dataTh['dB(GainLHCP) [] - F=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
G_LHCP = np.max(gainTh_P)
gainTh_P -= G_RHCP#np.max(gainTh_P)
print('Ganho com híbrida (dBi):', G_RHCP)

# Plotar o primeiro gráfico
axs[0].plot(angTh, gainTh_T, color='red', label = 'Simulação')
axs[0].set_title('Amplitude do Campo Elétrico RHCP')
axs[0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0].grid(True)
axs[0].set_theta_zero_location('N')
axs[0].set_theta_direction(-1)
axs[0].set_rlim(-30,0)
axs[0].set_thetagrids(angles)
axs[0].set_rlabel_position(45)

# Plotar o segundo gráfico
axs[1].plot(angTh, gainTh_P, color='red', label = 'Simulação')
axs[1].set_title('Amplitude do Campo Elétrico LHCP')
axs[1].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[1].grid(True)
axs[1].set_theta_zero_location('N')
axs[1].set_theta_direction(-1)
axs[1].set_rlim(-30,0)
axs[1].set_thetagrids(angles)
axs[1].set_rlabel_position(45)

handles, figlabels = axs[0].get_legend_handles_labels()
fig.legend(handles, figlabels, loc='lower center', ncol=1)
figures.append(fig)

plt.tight_layout()
# plt.show()

# Salvamento das figuras
for i, fig in enumerate(figures):
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.eps'), format = 'eps')
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.png'))