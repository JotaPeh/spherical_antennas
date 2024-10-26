from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import partial
import time
import warnings
import plotly.graph_objects as go
import pandas as pd

inicio_total = time.time()
show = 0

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
# phi1c = -58 * dtr
# phi2c = 238 * dtr

deltatheta1 = h/a       # Largura angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = h/a       # Largura angular 2 do campo de franjas e da abertura polar (rad)
# theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2 # Ângulos polares físicos do patch (rad)
deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)
# phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi             # Ângulos azimutais físicos do patch (rad)

# Coordenadas dos alimentadores coaxiais (rad)
# thetap = [90.0 * dtr, 95.0 * dtr]  
# phip = [82.4 * dtr, 90.0 * dtr]
# probes = len(thetap)    # Número de alimentadores
df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
er = 2.55               # Permissividade relativa do substrato dielétrico
es = e0 * er            # Permissividade do substrato dielétrico (F/m)
# DphiJ = [np.exp(1.5)*df/(2*a*np.sin(t)) for t in thetap]     # Comprimento angular do modelo da ponta de prova (rad)

# Outras variáveis
tgdel = 0.0022          # Tangente de perdas
sigma = 5.8e50          # Condutividade elétrica dos condutores (S/m)
Z0 = 50                 # Impedância intrínseca (ohm)

# Funções associadas de Legendre para |z| < 1
def legP(z, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if u.is_integer():
            return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
        else:
            return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

def legQ(z, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

################################
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
    
test = 0
if test:
    show = 1
    flm_des = 1575.42e6 # Hz, Banda L1
    Klm = 2*np.pi*flm_des*np.sqrt(u0*es)

    # Encontrar comprimento em Theta: Modo 1, 0
    rootsTheta = Theta_find(Klm)
    if show:
        print('Comprimentos em Theta (em graus):')
        print(rootsTheta / dtr)
        print('\nMeio comprimento de onda e Comprimento do patch em Theta no Equador:')
        print(c/(2 * flm_des * np.sqrt(er)), b*rootsTheta[0]-b*h/a,'\n\n')

    # Encontrar comprimento em Phi: Modo 0, 1
    rootsPhi = Phi_find(Klm, rootsTheta[0])
    if show:
        print('Comprimentos em Phi (em graus):')
        print(rootsPhi / dtr)
        print('\nMeio comprimento de onda e Comprimento do patch em Phi:')
        print(c/(2 * flm_des * np.sqrt(er)), b*rootsPhi[-1]-b*h/a)

    if show:
        print('\nDimensões: ', rootsTheta[0]/dtr, '° e ', rootsPhi[-1]/dtr, '°')
    theta1c, theta2c = np.pi/2 - rootsTheta[0]/2, np.pi/2 + rootsTheta[0]/2
    phi1c, phi2c = np.pi/2 - rootsPhi[-1]/2, np.pi/2 + rootsPhi[-1]/2

    Lambda = np.linspace(-0.1, 32, 64) # Domínio de busca, o professor usou a janela de 0.1
    rootsLambda = []

    def Equation(l, theta1f, theta2f, m):
        u = m * np.pi / (phi2c - phi1c)
        return np.where(np.abs(np.cos(theta1f)) < 1 and np.abs(np.cos(theta2f)) < 1, DP(theta1f,l,u)*DQ(theta2f,l,u)-DQ(theta1f,l,u)*DP(theta2f,l,u) , 1)

    def root_find(m):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots = []
            Wrapper = partial(Equation, theta1f = theta1c, theta2f = theta2c, m = m)
            k = 0
            for i in range(len(Lambda)-1):
                if Wrapper(Lambda[i]) * Wrapper(Lambda[i+1]) < 0:
                    root = root_scalar(Wrapper, bracket=[Lambda[i], Lambda[i+1]], method='bisect')
                    roots.append(root.root)
                    k += 1
            
            # print('all roots: ', roots)
            # Remoção de raízes inválidas
            k = 0
            for r in roots:
                # if round(r-roots[0], 6) % 1 == 0 and n != 0:
                if r < m * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
                    k += 1
            if show:
                print("n =", m, ":", [round(r, 5) for r in roots[k:k+5]], ', mu =', m * np.pi / (phi2c - phi1c), "\n\n")
            rootsLambda.append(roots[k:k+5])

    if show:
        print('\n\n\nRaízes lambda:')
    for m in range(0,5):
        root_find(m)

    flms = []
    if show:
        print('\nFrequências:')
    for r in rootsLambda:
        flms.append([round(p,5) for p in 1e-9*np.sqrt([x*(x + 1) for x in r])/(2*np.pi*a_bar*np.sqrt(er*e0*u0))])
        if show:
            print([round(p,5) for p in 1e-9*np.sqrt([x*(x + 1) for x in r])/(2*np.pi*a_bar*np.sqrt(er*e0*u0))])

else:
    # Projeto iterativo
    flm_des = 1575.42e6 # Hz, Banda L1 # Frequência de operação
    Kdes = 2*np.pi*flm_des*np.sqrt(u0*e0)

    def RLC(f, Kex, L, U, thetap, phip, Dphip, tgef, Dtheta, Dphi):
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        Ilm = spi.quad(partial(R2, l = L, m = U, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        # Auxiliar alpha
        alpha = h * (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) * \
            (R(np.cos(thetap), L, U, theta1c = theta1c) * np.cos(U * (phip-phi1c)) * np.sinc(U * Dphip / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
        if U != 0:
            alpha = 2 * alpha
        
        # Componentes RLC
        RLM = alpha/(2 * np.pi * f * tgef)
        CLM = 1/alpha
        LLM = alpha/(2 * np.pi * Kex / (2*np.pi*np.sqrt(u0*es)))**2
        return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

    def Z(f, k01, k10, thetap, phip, Dphip, tgef, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        k = (2 * np.pi * f) * np.sqrt(es * u0) 
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, k01, L01, M01, thetap, phip, Dphip, tgef, Dtheta, Dphi) + RLC(f, k10, L10, M10, thetap, phip, Dphip, tgef, Dtheta, Dphi) + 1j*Xp

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

    def tgef(k):
        # Qdie = 2 * np.pi * f * es / sigma_die # + Perdas de irradiação = 1/tgdel
        Rs = np.sqrt(k * np.sqrt(u0/es) / (2 * sigma))
        Qc = k * np.sqrt(u0/es) * h / (2 * Rs)
        Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)
        return tgdel + 1/Qc

    def tg10(k10, Dtheta, Dphi):
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        K0 = k10 / np.sqrt(er)

        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
        IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
        IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
        IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2
        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S10 = np.sum((((phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) ** 2 / (4*S_lm)) * (np.abs(b*(IdP_1+IdP_2)/dH2_dr)**2 + Mm**2 * np.abs((IpP_1+IpP_2)/(K0*H2))**2))

        Q10 = (np.pi / 12) * k10 * np.sqrt(er) * (b**3 - a**3) * I10 * Dphi / (np.abs(R(np.cos(theta1c),L10,M10,theta1c))**2 * S10)
        return tgef(k10) + 1/Q10
    
    def tg01(k01, Dtheta, Dphi): # Montar expressão de tg01 e tg10
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        K0 = k01 / np.sqrt(er)

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        I_th, I_dth = Integrais(theta1, theta2)
        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S01 = np.sum(((deltaPhi * np.sinc(Mm*deltaPhi/(2*np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2)) ** 2 / (4*S_lm)) * (Mm**2 * np.abs(b*I_th/dH2_dr)**2 + np.abs(I_dth/(K0*H2))**2))

        Q01 = (np.pi / 96) * k01 * np.sqrt(er) * (b**3 - a**3) * I01 * Dphi / (np.abs(R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c))**2 * S01)
        return tgef(k01) + 1/Q01

    def S(Kdes, Dtheta, Dphi):
        # L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi # Manter?
        # L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
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
            Sthv = np.sum(((1j ** Ml) * (-Kdes * b * sp.lpmn(m, l, 0)[1] * (IdP_1 + IdP_2) / (dH2_dr) \
                + 1j * Mm**2 * sp.lpmn(m, l, 0)[0] * (IpP_1 + IpP_2) / H2 ) * \
                (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) / S_lm))
            Sphh = np.sum(((1j ** Ml) * (Kdes * Mm**2 * b * I_th * sp.lpmn(m, l, 0)[0] / dH2_dr \
                + 1j * sp.lpmn(m, l, 0)[1] * -I_dth / H2 ) * \
                2 * Dphic * np.sinc(Mm*Dphic/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+Dphic)/2) / S_lm))

        Sr = Sthv/Sphh
        return Sr

    def Probe(k01, k10, Kef, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        thetap = np.linspace(theta1, theta2, 100)
        # print(thetap)

        # Dphip = np.exp(1.5)*df/(2*a*np.sin(thetap))

        V = (I01 * R(np.cos(theta1c),L10,M10,theta1c))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c)) * (Kef**2 - k01**2)/(Kef**2 - k10**2) * S(np.real(Kef), Dtheta, Dphi)
        # print(np.angle(V*1j))
        phip = theta1c + (1/M01)*np.arccos(R(np.cos(thetap),L10,M10,theta1c) / R(np.cos(thetap),L01,M01,theta1c)*np.abs(V))
        return phip, thetap
    
    def RA(k01, k10, Kef, Dtheta, Dphi, thetap, phip):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        Dphip = np.exp(1.5)*df/(2*a*np.sin(thetap))
        

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
 
        V = (I01 * R(np.cos(theta1c),L10,M10,theta1c))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c)) * (Kef**2 - k01**2)/(Kef**2 - k10**2) * S(np.real(Kef), Dtheta, Dphi)
        
        return (R(np.cos(thetap),L10,M10,theta1c) / (R(np.cos(thetap),L01,M01,theta1c) * np.cos(M01*(phip - phi1c)) * np.sinc(M01*Dphip/2))) * V
    
    # Início
    def imZin(p, Analysis = 0, Geometry = 0):
        k01 = k10 = 2*np.pi*flm_des*np.sqrt(u0*es)
        Dtheta = Theta_find(k10)[0]
        Dphi = Phi_find(k01, Dtheta)[-1]
        tgef = (tg10(k10, Dtheta, Dphi) + tg01(k01, Dtheta, Dphi))/2
    
        epsilon = 1
        tol = 1e-4
        while epsilon > tol:
            # print('k10 e k01: ', k01, k10)

            kl = k10 + p*(k01 - k10)
            kll = kl*tgef/2
            phaseK = np.pi/2 - np.angle(S(kl, Dtheta, Dphi))
            # print('phaseS: ', np.angle(S(kl, Dtheta, Dphi))/dtr,'phaseK: ', phaseK/dtr)

            k01prev = k01
            k10prev = k10

            # Supondo k01 > k10
            k10 = kl - (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*(1-p))
            k01 = kl + (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*p)
            
            Dtheta = Theta_find(k10)[0]
            Dphi = Phi_find(k01, Dtheta)[-1]
            tgef = (1-p)*tg10(k10, Dtheta, Dphi) + p*tg01(k01, Dtheta, Dphi)
            # print('tg: ', tg10(k10, Dtheta, Dphi),tg01(k01, Dtheta, Dphi))

            epsilon = np.max([np.abs(k01prev - k01), np.abs(k10prev - k10)])

        print('k01 e k10 final: ', k01, k10)

        Ph, Th = Probe(k01, k10, kl-1j*kll, Dtheta, Dphi)
        Phtheta = interp1d(Th, Ph)

        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2

        if  Analysis:
            fig, ax = plt.subplots()
            rect = patches.Rectangle((phi1c/dtr, theta1c/dtr), phi2c/dtr - phi1c/dtr, theta2c/dtr - theta1c/dtr, color='#B87333', zorder=0)
            ax.add_patch(rect)
            ax.plot(Ph/dtr, Th/dtr)
            ax.plot(np.linspace(phi1c, phi2c, 100)/dtr, np.linspace(theta1c, theta2c, 100)/dtr, dashes=[4, 4])
            ax.set_title('Lugar geométrico da ponta de prova')
            ax.grid(False)
            ax.set_xlim(phi1c/dtr - h/(a*dtr), phi2c/dtr + h/(a*dtr))
            ax.set_ylim(theta1c/dtr - h/(a*dtr), theta2c/dtr + h/(a*dtr))
            plt.gca().invert_yaxis()
            plt.show()
            # Lugar geométrico das pontas de prova
            # Posicionar ponta de prova em Zin = 50

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=15, azim=100)

            # Crie a esfera
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = b * np.outer(np.cos(u), np.sin(v))
            y = b * np.outer(np.sin(u), np.sin(v))
            z = b * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z, color='white', alpha = 0.5, zorder=0)

            # Crie o retângulo na superfície da esfera
            phi = np.linspace(phi1c, phi2c, 50)
            theta = np.linspace(theta1c, theta2c, 50)
            phi, theta = np.meshgrid(phi, theta)

            x_rect = b * np.cos(phi) * np.sin(theta)
            y_rect = b * np.sin(phi) * np.sin(theta)
            z_rect = b * np.cos(theta)

            ax.plot_surface(x_rect, y_rect, z_rect, color='#B87333', zorder=1)

            x_curve = b * np.cos(Ph) * np.sin(Th)
            y_curve = b * np.sin(Ph) * np.sin(Th)
            z_curve = b * np.cos(Th)

            ax.plot(x_curve, y_curve, z_curve, color='blue', linewidth=4, zorder=2)

            # Ajuste a visualização
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

            # fig = plt.figure()
            # phi_proj = np.linspace(phi1c, phi2c, 50)
            # theta_proj = np.linspace(theta1c, theta2c, 50)
            # phi_proj, theta_proj = np.meshgrid(phi_proj, theta_proj)

            # x_rect_proj = b * np.cos(phi_proj) * np.sin(theta_proj)
            # z_rect_proj = b * np.cos(theta_proj)

            # # Projeção da curva no plano XZ
            # x_curve_proj = b * np.cos(Ph) * np.sin(Th)
            # z_curve_proj = b * np.cos(Th)

            # # Plot 2D do retângulo e da curva projetados no plano XZ
            # plt.fill(x_rect_proj, z_rect_proj, color='#B87333', label='Retângulo Projetado',alpha = 0.5)
            # plt.plot(x_curve_proj, z_curve_proj, color='blue', label='Curva Projetada')

            plt.show()


        Zinth = Z(flm_des, k01, k10, Th, Ph, np.exp(1.5)*df/(2*a_bar*np.sin(Th)), tgef, Dtheta, Dphi)

        def Z50(theta):
            return np.real(Z(flm_des, k01, k10, theta, Phtheta(theta), np.exp(1.5)*df/(2*a_bar*np.sin(theta)), tgef, Dtheta, Dphi))-50
        root = root_scalar(Z50, bracket=[np.pi/2, theta2c-0.02], method='bisect')

        thetap = np.array(root.root)
        # print(thetap/dtr)

        if Analysis:
            plt.figure()
            plt.plot(Th/dtr, np.real(Zinth))
            plt.plot(Th/dtr, np.imag(Zinth))
            plt.title('Impedância')
            plt.grid(True)
            plt.show()

            axial =  RA(k01, k10, kl-1j*kll, Dtheta, Dphi, thetap, Phtheta(thetap))
            print('Razão axial: ', axial, ' Módulo:', np.abs(axial),' Módulo (dB):', np.abs(20*np.log10(np.abs(axial))), 'Fase: ', np.angle(axial)/dtr)

        Zin = Z(flm_des, k01, k10, thetap, Phtheta(thetap), np.exp(1.5)*df/(2*a_bar*np.sin(thetap)), tgef, Dtheta, Dphi)
        # print('Zin = ', Zin)
        
        if Geometry == 1:
            return Dphi, Dtheta, thetap, Phtheta(thetap), Zin
        print(p, np.imag(Zin))
        return np.imag(Zin)

    Wrapper = partial(imZin, Analysis = 0, Geometry = 0)

    project = 0

    if project:
        inicio_projeto = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = root_scalar(Wrapper, bracket=[0.5, 0.7], method='bisect', xtol = 1e-4)
            Dphi, Dtheta, thetap, phip, Zin = imZin(np.array(root.root), Geometry = 1)
            print(Dphi/dtr, Dtheta/dtr, thetap/dtr, phip/dtr, Zin)
        fim_projeto = time.time()
        print("Tempo de projeto: ", fim_projeto-inicio_projeto,' s')
        print('p = ', np.array(root.root))
    else:
        Dphi, Dtheta, thetap, phip, Zin = imZin(0.58173828125, Analysis = 1, Geometry = 1)

        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi

        print('theta1c, theta2c, phi1c, phi2c, thetap, phip, Zin: ')
        print(theta1c/dtr, theta2c/dtr, phi1c/dtr, phi2c/dtr, thetap/dtr, phip/dtr, Zin)
        print('phi1, phi2, theta1, theta2: ')
        print(phi1/dtr,phi2/dtr,theta1/dtr,theta2/dtr)

        fig, ax = plt.subplots()
        rect = patches.Rectangle((phi1/dtr, theta1c/dtr), phi2/dtr - phi1/dtr, theta2c/dtr - theta1c/dtr, color='gray', zorder=1)
        rect2 = patches.Rectangle((phi1c/dtr, theta1/dtr), phi2c/dtr - phi1c/dtr, theta2/dtr - theta1/dtr, color='gray', zorder=1)
        rect3 = patches.Rectangle((phi1/dtr, theta1/dtr), phi2/dtr - phi1/dtr, theta2/dtr - theta1/dtr, color='#B87333', zorder=2)
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.scatter(phip/dtr, thetap/dtr, s=50, color='orange', zorder=5, label='Ponto 1')  # Ponto 1: círculo maior e azul
        ax.scatter(phip/dtr, thetap/dtr, s=100, color='red', zorder=4, label='Ponto 2')   # Ponto 2: círculo ainda maior e vermelho
        ax.plot(np.linspace(phi1, phi2, 100)/dtr, np.linspace(theta1, theta2, 100)/dtr, dashes=[4, 4], zorder=3)
        ax.set_title('Geometria do patch')
        ax.grid(False)
        ax.set_xlim(phi1c/dtr - 5*h/(a*dtr), phi2c/dtr + 5*h/(a*dtr))
        ax.set_ylim(theta1c/dtr - 5*h/(a*dtr), theta2c/dtr + 5*h/(a*dtr))
        ax.set_facecolor('lightgray')
        plt.gca().invert_yaxis()
        plt.show()