from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import partial
import time
import warnings
import os

inicio_total = time.time()
show = 0

# Constantes gerais
dtr = np.pi/180         # Graus para radianos (rad/°)
e0 = 8.854e-12          # (F/m)
u0 = np.pi*4e-7         # (H/m)
c = 1/np.sqrt(e0*u0)    # Velocidade da luz no vácuo (m/s)
gamma = 0.577216        # Constante de Euler-Mascheroni
eps = 1e-5              # Limite para o erro numérico

# Geometria da cavidade
a = 100e-3              # Raio da esfera de terra (m)
h = 1.524e-3            # Espessura do substrato dielétrico (m)
a_bar = a + h/2         # Raio médio da esfera de terra (m)
b = a + h               # Raio exterior do dielétrico (m)

deltatheta1 = h/a       # Largura angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = h/a       # Largura angular 2 do campo de franjas e da abertura polar (rad)
deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)

# Coordenadas dos alimentadores coaxiais (rad)
df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
er = 2.55               # Permissividade relativa do substrato dielétrico
es = e0 * er            # Permissividade do substrato dielétrico (F/m)

# Outras variáveis
tgdel = 0.0022          # Tangente de perdas
sigma = 5.8e50          # Condutividade elétrica dos condutores (S/m)
Z0 = 50                 # Impedância intrínseca (ohm)

output_folder = 'Resultados/Patch'
os.makedirs(output_folder, exist_ok=True)
figures = []

# Funções associadas de Legendre para |z| < 1
def legP(z, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(u, list):
            u = np.array(u)
            return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)
        u = float(u)
        if u.is_integer():
            return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
        else:
            return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

def legQ(z, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(u, list):
            u = np.array(u)
            return np.where(np.abs(z) < 1, np.pi * (np.cos(u*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
        - sp.gamma(l+u+1) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / (sp.gamma(1+u) * sp.gamma(l-u+1))) / (2 * np.sin(u*np.pi)), 1)

        u = float(u)
        if u.is_integer():
            return np.where(np.abs(z) < 1, np.pi * (np.cos((u+l)*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
        - np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1+z)/2)  / sp.gamma(1-u)) / (2 * np.sin((u+l)*np.pi)), 1)
        else:
            return np.where(np.abs(z) < 1, np.pi * (np.cos(u*np.pi) * np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u) \
        - sp.gamma(l+u+1) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / (sp.gamma(1+u) * sp.gamma(l-u+1))) / (2 * np.sin(u*np.pi)), 1)

# Derivadas
def DP(theta, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.cos(theta)
        return l*legP(z, l, u)/np.tan(theta) - (l+u) * legP(z, l-1, u) / np.sin(theta)

def DQ(theta, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = np.cos(theta)
        return l*legQ(z, l, u)/np.tan(theta) - (l+u) * legQ(z, l-1, u) / np.sin(theta)

#########################################################################################################################################################
# Funções auxiliares de busca de raízes
def EquationCircTheta(Dtheta, Dphi, m, K):
    theta1, theta2 = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
    Lamb = (np.sqrt(1+4*a_bar**2 * K**2)-1)/2
    u = m * np.pi / Dphi
    return np.where(np.abs(np.cos(theta1)) < 1 and np.abs(np.cos(theta2)) < 1, DP(theta1,Lamb,u)*DQ(theta2,Lamb,u)-DQ(theta1,Lamb,u)*DP(theta2,Lamb,u) , 1)

def EquationCircPhi(Dphi, Dtheta, m, K):
    theta1, theta2 = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
    Lamb = (np.sqrt(1+4*a_bar**2 * K**2)-1)/2
    u = m * np.pi / Dphi
    return np.where(np.abs(np.cos(theta1)) < 1 and np.abs(np.cos(theta2)) < 1, DP(theta1,Lamb,u)*DQ(theta2,Lamb,u)-DQ(theta1,Lamb,u)*DP(theta2,Lamb,u) , 1)

def Theta_find(K, Dphi = 1):
    Thetas = np.linspace(0.1, 359.9, 601) * dtr # Domínio de busca
    roots = []
    Wrapper = partial(EquationCircTheta, Dphi = Dphi,  m = 0, K = K) # Modo TM^r_10 desejado
    k = 0
    for i in range(len(Thetas)-1):
        if Wrapper(Thetas[i]) * Wrapper(Thetas[i+1]) < 0:
            root = root_scalar(Wrapper, bracket=[Thetas[i], Thetas[i+1]], method='bisect')
            roots.append(root.root)
            k += 1

    return np.array(roots)

def Phi_find(K, Dtheta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Phis = np.linspace(0.1, 179.9, 351) * dtr # Domínio de busca
        roots = []
        Wrapper = partial(EquationCircPhi, Dtheta = Dtheta,  m = 1, K = K) # Modo TM^r_01 desejado
        k = 0
        for i in range(len(Phis)-1):
            if Wrapper(Phis[i]) * Wrapper(Phis[i+1]) < 0:
                root = root_scalar(Wrapper, bracket=[Phis[i], Phis[i+1]], method='bisect')
                roots.append(root.root)
                k += 1

        return np.array(roots)

#########################################################################################################################################################
# Projeto da antena com duas pontas
flm_des = 1575.42e6 # Banda L1
l = m = 35 # m <= l

Kdes = 2*np.pi*flm_des*np.sqrt(u0*e0)
Ml, Mm = np.meshgrid(np.arange(0,l+1), np.arange(0,m+1))
delm = np.ones(m+1)
delm[0] = 0.5

def RLC(f, Kex, L, U, thetap, phip, tgef, Dtheta, Dphi):
    Dphip = np.exp(1.5)*df/(2*a_bar*np.sin(thetap))
    theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
    phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
    Ilm = spi.quad(partial(R2, l = L, m = U, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

    alpha = h * (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) * \
        (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
    if U != 0:
        alpha = 2 * alpha
    
    RLM = alpha/(2 * np.pi * f * tgef)
    CLM = 1/alpha
    LLM = alpha/(2 * np.pi * Kex / (2*np.pi*np.sqrt(u0*es)))**2
    return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

def Z(f, klm, thetap, phip, tgef01, tgef10, Dtheta, Dphi):
    L01, M01 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, np.pi/Dphi
    L10, M10 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, 0
    k = (2 * np.pi * f) * np.sqrt(es * u0) 
    eta = np.sqrt(u0 / es)
    Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
    return RLC(f, klm, L01, M01, thetap, phip, tgef01, Dtheta, Dphi) + RLC(f, klm, L10, M10, thetap, phip, tgef10, Dtheta, Dphi) + 1j*Xp

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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    S_lm = (2 * Ml * (Ml+1) * sp.gamma(1+Ml+Mm)) / ((2*Ml+1) * sp.gamma(1+Ml-Mm))
    S_lm += (1-np.abs(np.sign(S_lm)))*eps

def Integrais(theta1, theta2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        I_th = np.array([[spi.quad(partial(Itheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
        I_dth = np.array([[spi.quad(partial(IDtheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
    return I_th, I_dth

def R(v,L,m, theta1c):
    return (DP(theta1c,L,m)*legQ(v,L,m) - DQ(theta1c,L,m)*legP(v,L,m)) * np.sin(theta1c)

def R2(v, l, m, theta1c):
    return R(v, l, m, theta1c)**2

def tgefTot(klm, mode, Dtheta, Dphi):
    Rs = np.sqrt(klm * np.sqrt(u0/es) / (2 * sigma))
    Qc = klm * np.sqrt(u0/es) * h / (2 * Rs)
    Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)
    
    if mode == '10':
        L10, M10 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        K0 = klm / np.sqrt(er)

        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
        IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
        IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
        IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2
        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S10 = (((phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) ** 2 / (S_lm)) * (np.abs(b*(IdP_1+IdP_2)/dH2_dr)**2 + Mm**2 * np.abs((IpP_1+IpP_2)/(K0*H2))**2)
        S10 = np.dot(delm, S10)
        
        Q10 = (np.pi / 3) * klm * np.sqrt(er) * (b**3 - a**3) * I10 * Dphi / (np.abs(R(np.cos(theta1c),L10,M10,theta1c))**2 * np.sum(S10))
        return tgdel + 1/Qc + 1/Q10
    elif mode == '01':
        L01, M01 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, np.pi/Dphi
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        K0 = klm / np.sqrt(er)

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        I_th, I_dth = Integrais(theta1, theta2)
        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S01 = ((deltaPhi * np.sinc(Mm*deltaPhi/(2*np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2)) ** 2 / (S_lm)) * (Mm**2 * np.abs(b*I_th/dH2_dr)**2 + np.abs(I_dth/(K0*H2))**2)
        S01 = np.dot(delm, S01)

        Q01 = (np.pi / 24) * klm * np.sqrt(er) * (b**3 - a**3) * I01 * Dphi / (np.abs(R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c))**2 * np.sum(S01))
        return tgdel + 1/Qc + 1/Q01

def S(Kdes, Dtheta, Dphi):
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
    I_th, I_dth = Integrais(theta1, theta2)
    dH2_dr = np.tile(schelkunoff2(l, Kdes * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, Kdes * b),(m + 1, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sthv = ((1j ** Ml) * (-Kdes * b * sp.lpmn(m, l, 0)[1] * (IdP_1 + IdP_2) / (dH2_dr) \
            + 1j * Mm**2 * sp.lpmn(m, l, 0)[0] * (IpP_1 + IpP_2) / H2 ) * \
            (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) / S_lm)
        Sthv = np.dot(delm, Sthv)
        Sphh = ((1j ** Ml) * (Kdes * Mm**2 * b * I_th * sp.lpmn(m, l, 0)[0] / dH2_dr \
            + 1j * sp.lpmn(m, l, 0)[1] * -I_dth / H2 ) * \
            2 * Dphic * np.sinc(Mm*Dphic/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+Dphic)/2) / S_lm)
        Sphh = np.dot(delm, Sphh)

    Sr = np.sum(Sthv)/np.sum(Sphh)
    return Sr

def RA(klm, freq, tgef01, tgef10, Dtheta, Dphi, thetap2, phip1, ratioI):
    L01, M01 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, np.pi/Dphi
    L10, M10 = (np.sqrt(1+4*a_bar**2 * klm**2)-1)/2, 0
    theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
    phi1c = np.pi/2 - Dphi/2
    Dphip = np.exp(1.5)*df/(2*a*np.sin(np.pi/2))
    Kef01 = 2*np.pi*freq*np.sqrt(u0*es*(1-1j*tgef01))
    Kef10 = 2*np.pi*freq*np.sqrt(u0*es*(1-1j*tgef10))

    I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
    I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

    V = (I01 * R(np.cos(theta1c),L10,M10,theta1c))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c)) * (Kef01**2 - klm**2)/(Kef10**2 - klm**2) * S(klm, Dtheta, Dphi)

    return ratioI * (R(np.cos(thetap2),L10,M10,theta1c) / (R(np.cos(np.pi/2),L01,M01,theta1c) * np.cos(M01*(phip1 - phi1c)) * np.sinc(M01*Dphip/(2*np.pi)))) * V

# Início do Projeto
def ProjCP2():
    klm = 2*np.pi*flm_des*np.sqrt(u0*es)
    Dtheta = Theta_find(klm)[0]
    Dphi = Phi_find(klm, Dtheta)[-1]
    tgef10 = tgefTot(klm, '10', Dtheta, Dphi)
    tgef01 = tgefTot(klm, '01', Dtheta, Dphi)

    thetap1 = phip2 = np.pi/2

    def Z01(phi):
        return np.real(Z(flm_des, klm, thetap1, phi, tgef01, tgef10, Dtheta, Dphi))-50
    root = root_scalar(Z01, bracket=[np.pi/2, np.pi/2+Dphi/2-0.02-h/a], method='bisect')
    phip1 = np.array(root.root)

    def Z10(theta):
        return np.real(Z(flm_des, klm, theta, phip2, tgef01, tgef10, Dtheta, Dphi))-50
    root = root_scalar(Z10, bracket=[np.pi/2, np.pi/2+Dtheta/2-0.02-h/a], method='bisect')
    thetap2 = np.array(root.root)
    
    RAb = RA(klm, flm_des, tgef01, tgef10, Dtheta, Dphi, thetap2, phip1, 1)
    ratioI = 1/np.abs(RAb) * np.exp(1j*(np.pi/2 - np.angle(RAb)))

    if show:
        print('RA (abs/angle): ', np.abs(RA(klm, flm_des, tgef01, tgef10, Dtheta, Dphi, thetap2, phip1, ratioI)), np.angle(RA(klm, flm_des, tgef01, tgef10, Dtheta, Dphi, thetap2, phip1, ratioI))/dtr, '\n')
        print('Z11: ',Z(flm_des, klm, thetap1, phip1, tgef01, tgef10, Dtheta, Dphi), '\nZ22: ', Z(flm_des, klm, thetap2, phip2, tgef01, tgef10, Dtheta, Dphi), '\n RA: ', RA(klm, flm_des, tgef01, tgef10, Dtheta, Dphi, thetap2, phip1, ratioI), '\n\n')

    return Dphi, Dtheta, thetap1, thetap2, phip1, phip2, ratioI

Dphi, Dtheta, thetap1, thetap2, phip1, phip2, ratioI = ProjCP2()

theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi

if show:
    print('theta1c, theta2c, phi1c, phi2c, thetap2, phip1, ratioI: ')
    print(theta1c/dtr, theta2c/dtr, phi1c/dtr, phi2c/dtr, thetap2/dtr, phip1/dtr, ratioI, '\n')
    print('thetap2, phip1, theta1, theta2, phi1, phi2: ')
    print(thetap2/dtr, phip1/dtr, theta1/dtr,theta2/dtr,phi1/dtr,phi2/dtr, '\n')
    print('ratioI (abs/angle): ', np.abs(ratioI), np.angle(ratioI)/dtr, '\n')

fig, ax = plt.subplots()
rect = patches.Rectangle((phi1/dtr, theta1c/dtr), phi2/dtr - phi1/dtr, theta2c/dtr - theta1c/dtr, color='gray', zorder=1)
rect2 = patches.Rectangle((phi1c/dtr, theta1/dtr), phi2c/dtr - phi1c/dtr, theta2/dtr - theta1/dtr, color='gray', zorder=1)
rect3 = patches.Rectangle((phi1/dtr, theta1/dtr), phi2/dtr - phi1/dtr, theta2/dtr - theta1/dtr, color='#B87333', zorder=2)
ax.add_patch(rect)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.scatter(phip1/dtr, np.pi/2/dtr, s=50, color='orange', zorder=5, label='Ponto 1')
ax.scatter(phip1/dtr, np.pi/2/dtr, s=100, color='red', zorder=4, label='Ponto 2')
ax.scatter(np.pi/2/dtr, thetap2/dtr, s=50, color='orange', zorder=5, label='Ponto 1')
ax.scatter(np.pi/2/dtr, thetap2/dtr, s=100, color='red', zorder=4, label='Ponto 2')
ax.text(phip1/dtr, (np.pi/2)/dtr - 1.3, '1', color='#006400', fontsize=12, ha='center', va='bottom')  # Número 1 em cima do ponto
ax.text(np.pi/2/dtr - 1, thetap2/dtr, '2', color='#006400', fontsize=12, ha='right', va='center')     # Número 2 à esquerda do ponto 2
ax.plot(np.linspace(phi1, phi2, 100)/dtr, np.array([np.pi/2] * 100)/dtr, dashes=[4, 4], zorder=3)
ax.plot(np.array([np.pi/2] * 100)/dtr, np.linspace(theta1, theta2, 100)/dtr, dashes=[4, 4], zorder=3)
ax.set_title('Geometria do patch')
ax.grid(False)
ax.set_xlim(phi1c/dtr - 5*h/(a*dtr), phi2c/dtr + 5*h/(a*dtr))
ax.set_ylim(theta1c/dtr - 5*h/(a*dtr), theta2c/dtr + 5*h/(a*dtr))
ax.set_facecolor('lightgray')
plt.ylabel(r'$\theta$'+' (°)' )
plt.xlabel(r'$\varphi$'+' (°)' )
plt.gca().invert_yaxis()

# Exportar resultado
parametros = f"Dtheta = {Dtheta}\nDphi = {Dphi}\nthetap1 = {thetap1}\nthetap2 = {thetap2}\nphip1 = {phip1}\nphip2 = {phip2}\nre_ratioI = {np.real(ratioI)}\nim_ratioI = {np.imag(ratioI)}\nf = {flm_des}"
out_dir = 'Resultados/Param'
out_file = os.path.join(out_dir, 'CP_2P_Param.txt')
os.makedirs(out_dir, exist_ok=True)
with open(out_file, 'w') as f:
    f.write(parametros)
if show:
    print(f"Arquivo salvo em {out_file}")

fig.savefig(os.path.join(output_folder, 'figure_CP2.eps'), format='eps')
fig.savefig(os.path.join(output_folder, 'figure_CP2.png'))
fim_total = time.time()
print("Tempo total para o fim do código: ", fim_total - inicio_total, "segundos\n")

if show:
    plt.show()