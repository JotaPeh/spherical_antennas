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
import os

inicio_total = time.time()
test = 0
project = 0
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

output_folder = 'Resultados/Sintese_CP'
os.makedirs(output_folder, exist_ok=True)
figures = []

# Funções associadas de Legendre para |z| < 1
def legP(z, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = float(u)
        if u.is_integer():
            return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
        else:
            return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

def legQ(z, l, u):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

if test:
    #########################################################################################################################################################
    # Testes das funções de busca de raízes
    flm_des = 1575.42e6 # Hz, Banda L1

    show = 1
    Klm = 2*np.pi*flm_des*np.sqrt(u0*es)

    # Encontrar comprimento em Theta: Modo 1, 0
    rootsTheta = Theta_find(Klm)
    if show:
        print('Comprimentos em Theta (em graus):')
        print(rootsTheta / dtr)
        print('\nMeio comprimento de onda e Comprimento do patch em Theta no Equador:')
        print(c/(2 * flm_des * np.sqrt(er)), b*rootsTheta[0],'\n\n')

    # Encontrar comprimento em Phi: Modo 0, 1
    rootsPhi = Phi_find(Klm, rootsTheta[0])
    if show:
        print('Comprimentos em Phi (em graus):')
        print(rootsPhi / dtr)
        print('\nMeio comprimento de onda e Comprimento do patch em Phi:')
        print(c/(2 * flm_des * np.sqrt(er)), b*rootsPhi[-1])

    if show:
        print('\n\nDimensões: ', rootsTheta[0]/dtr, '° e ', rootsPhi[-1]/dtr, '°')
    theta1c, theta2c = np.pi/2 - rootsTheta[0]/2, np.pi/2 + rootsTheta[0]/2
    phi1c, phi2c = np.pi/2 - rootsPhi[-1]/2, np.pi/2 + rootsPhi[-1]/2

    Lambda = np.linspace(-0.1, 32, 64)
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

            # Remoção de raízes inválidas
            k = 0
            for r in roots:
                if r < m * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
                    k += 1
            rootsLambda.append(roots[k:k+5])

    # Raízes lambda
    for m in range(0,5):
        root_find(m)

    flms = []
    if show:
        print('\n\nFrequências:')
    for r in rootsLambda:
        flms.append([round(p,5) for p in 1e-9*np.sqrt([x*(x + 1) for x in r])/(2*np.pi*a_bar*np.sqrt(er*e0*u0))])
        if show:
            print([round(p,5) for p in 1e-9*np.sqrt([x*(x + 1) for x in r])/(2*np.pi*a_bar*np.sqrt(er*e0*u0))])

else:
    #########################################################################################################################################################
    # Projeto iterativo
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

    def Z(f, k01, k10, thetap, phip, tgef, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        k = (2 * np.pi * f) * np.sqrt(es * u0) 
        eta = np.sqrt(u0 / es)
        Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
        return RLC(f, k01, L01, M01, thetap, phip, tgef, Dtheta, Dphi) + RLC(f, k10, L10, M10, thetap, phip, tgef, Dtheta, Dphi) + 1j*Xp

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

    def Probe(k01, k10, Kef, Dtheta, Dphi):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c = np.pi/2 - Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]

        thetap = np.linspace(theta1, theta2, 101)

        V = (I01 * R(np.cos(theta1c),L10,M10,theta1c))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c)) * (Kef**2 - k01**2)/(Kef**2 - k10**2) * S(np.real(Kef), Dtheta, Dphi)
        phip = phi1c + np.arccos(R(np.cos(thetap),L10,M10,theta1c) / R(np.cos(thetap),L01,M01,theta1c) * np.abs(V)) / M01
        
        return phip, thetap

    def RA(k01, k10, Kef, Dtheta, Dphi, thetap, phip):
        L01, M01 = (np.sqrt(1+4*a_bar**2 * k01**2)-1)/2, np.pi/Dphi
        L10, M10 = (np.sqrt(1+4*a_bar**2 * k10**2)-1)/2, 0
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c = np.pi/2 - Dphi/2
        Dphip = np.exp(1.5)*df/(2*a*np.sin(thetap))
        

        I01 = spi.quad(partial(R2, l = L01, m = M01, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
        I10 = spi.quad(partial(R2, l = L10, m = M10, theta1c = theta1c),np.cos(theta2c),np.cos(theta1c))[0]
 
        V = (I01 * R(np.cos(theta1c),L10,M10,theta1c))/(2 * I10 * R(np.cos((theta1c+theta2c)/2),L01,M01,theta1c)) * (Kef**2 - k01**2)/(Kef**2 - k10**2) * S(np.real(Kef), Dtheta, Dphi)

        return (R(np.cos(thetap),L10,M10,theta1c) / (R(np.cos(thetap),L01,M01,theta1c) * np.cos(M01*(phip - phi1c)) * np.sinc(M01*Dphip/(2*np.pi)))) * V
    
    # Início do Projeto
    def imZin(p, Analysis = 0):
        k01 = k10 = 2*np.pi*flm_des*np.sqrt(u0*es)
        Dtheta = Theta_find(k10)[0]
        Dphi = Phi_find(k01, Dtheta)[-1]
        tgef = (tgefTot(k10, '10', Dtheta, Dphi) + tgefTot(k01, '01', Dtheta, Dphi))/2
    
        epsilon = 1
        tol = 1e-4
        steps = 0
        while epsilon > tol:
            steps += 1
            kl = k10 + p*(k01 - k10)
            kll = kl*tgef/2
            phaseK = np.pi/2 - np.angle(S(kl, Dtheta, Dphi))

            k01prev = k01
            k10prev = k10

            # Supondo-se k01 > k10
            k10 = kl - (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*(1-p))
            k01 = kl + (-1/np.tan(phaseK) + np.sqrt((1/np.tan(phaseK))**2 + 4*(1-p)*p) ) * kll / (2*p)
            
            Dtheta = Theta_find(k10)[0]
            Dphi = Phi_find(k01, Dtheta)[-1]
            tgef = (1-p)*tgefTot(k10, '10', Dtheta, Dphi) + p*tgefTot(k01, '01', Dtheta, Dphi)

            epsilon = np.max([np.abs(k01prev - k01), np.abs(k10prev - k10)])

        if not Analysis:
            print('k01 e k10 final: ', k01, k10, '\n')
            print("Passos para a convergência dos números de onda: ", steps, '\n')

        Ph, Th = Probe(k01, k10, kl-1j*kll, Dtheta, Dphi)

        Phtheta = interp1d(Th, Ph)

        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2

        def Z50(theta):
            return np.real(Z(flm_des, k01, k10, theta, Phtheta(theta), tgef, Dtheta, Dphi))-50
        root = root_scalar(Z50, bracket=[np.pi/2, theta2c-0.02], method='bisect')

        thetap = np.array(root.root)

        Zin = Z(flm_des, k01, k10, thetap, Phtheta(thetap), tgef, Dtheta, Dphi)

        if  Analysis:
            # Representações gráficas auxiliares do patch e do lugar geométrico projetados
            fig, ax = plt.subplots()
            rect = patches.Rectangle((phi1c/dtr, theta1c/dtr), phi2c/dtr - phi1c/dtr, theta2c/dtr - theta1c/dtr, color='#B87333', zorder=0)
            ax.add_patch(rect)
            ax.plot(Ph/dtr, Th/dtr)
            ax.plot(np.linspace(phi1c, phi2c, 100)/dtr, np.linspace(theta1c, theta2c, 100)/dtr, dashes=[4, 4])
            ax.set_title('Lugar geométrico da ponta de prova')
            plt.ylabel(r'$\theta$'+'(°)' )
            plt.xlabel(r'$\varphi$'+'(°)' )
            ax.grid(False)
            ax.set_xlim(phi1c/dtr - h/(a*dtr), phi2c/dtr + h/(a*dtr))
            ax.set_ylim(theta1c/dtr - h/(a*dtr), theta2c/dtr + h/(a*dtr))
            plt.gca().invert_yaxis()
            figures.append(fig)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=15, azim=100)

            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = b * np.outer(np.cos(u), np.sin(v))
            y = b * np.outer(np.sin(u), np.sin(v))
            z = b * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z, color='white', alpha = 0.5, zorder=2)

            phi = np.linspace(phi1c, phi2c, 50)
            theta = np.linspace(theta1c, theta2c, 50)
            phi, theta = np.meshgrid(phi, theta)

            x_rect = b * np.cos(phi) * np.sin(theta)
            y_rect = b * np.sin(phi) * np.sin(theta)
            z_rect = b * np.cos(theta)

            ax.plot_surface(x_rect, y_rect, z_rect, color='#B87333', zorder=1)

            x_curve = (b+h*2) * np.cos(Ph) * np.sin(Th)
            y_curve = (b+h*2) * np.sin(Ph) * np.sin(Th)
            z_curve = (b+h*2) * np.cos(Th)

            ax.plot(x_curve, y_curve, z_curve, color='blue', linewidth=4, zorder=0)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            figures.append(fig)
            
            Zinth = Z(flm_des, k01, k10, Th, Ph, tgef, Dtheta, Dphi)
            
            fig = plt.figure()
            plt.plot(Th/dtr, np.real(Zinth), label = 'Re'+r'($Z_{11}$)')
            plt.plot(Th/dtr, np.imag(Zinth), label = 'Im'+r'($Z_{11}$)')
            plt.title('Impedância '+r'$Z_{11}$')
            plt.xlabel(r'$\theta$'+'(°)' )
            plt.ylabel('Impedância '+r'($\Omega$)')
            plt.legend()
            plt.grid(True)
            figures.append(fig)

            axial =  RA(k01, k10, kl-1j*kll, Dtheta, Dphi, thetap, Phtheta(thetap))

            if show:
                print('Razão axial: ', axial, ' Módulo:', np.abs(axial),' Módulo (dB):', np.abs(20*np.log10(np.abs(axial))), 'Fase: ', np.angle(axial)/dtr, '\n')

            return Dphi, Dtheta, thetap, Phtheta(thetap), Zin
        if show and not project:
            print(p, np.imag(Zin), '\n')
        return np.imag(Zin)

    if project:
        Wrapper = partial(imZin, Analysis = 0)
        inicio_projeto = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            root = root_scalar(Wrapper, bracket=[0.5, 0.7], method='bisect', xtol = 1e-4)
            print('Convergência após ', root.iterations, ' passos\n')
            Dphi, Dtheta, thetap, phip, Zin = imZin(np.array(root.root), Analysis = 1)
            print('Dphi, Dtheta, thetap, phip, Zin: ')
            print(Dphi/dtr, Dtheta/dtr, thetap/dtr, phip/dtr, Zin)
        fim_projeto = time.time()
        print("\nTempo de projeto: ", fim_projeto-inicio_projeto,' s \n')
        p = np.array(root.root)
        print('p = ', p, '\n')
    else:
        p = 0.5891601562500001 # Resultado do projeto

        Dphi, Dtheta, thetap, phip, Zin = imZin(p, Analysis = 1)

        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi

        if show:
            print('theta1c, theta2c, phi1c, phi2c, thetap, phip, Zin: ')
            print(theta1c/dtr, theta2c/dtr, phi1c/dtr, phi2c/dtr, thetap/dtr, phip/dtr, Zin, '\n')
            print('thetap, phip, theta1, theta2, phi1, phi2: ')
            print(thetap/dtr, phip/dtr, theta1/dtr,theta2/dtr,phi1/dtr,phi2/dtr, '\n')

        fig, ax = plt.subplots()
        rect = patches.Rectangle((phi1/dtr, theta1c/dtr), phi2/dtr - phi1/dtr, theta2c/dtr - theta1c/dtr, color='gray', zorder=1)
        rect2 = patches.Rectangle((phi1c/dtr, theta1/dtr), phi2c/dtr - phi1c/dtr, theta2/dtr - theta1/dtr, color='gray', zorder=1)
        rect3 = patches.Rectangle((phi1/dtr, theta1/dtr), phi2/dtr - phi1/dtr, theta2/dtr - theta1/dtr, color='#B87333', zorder=2)
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.scatter(phip/dtr, thetap/dtr, s=50, color='orange', zorder=5, label='Ponto 1')
        ax.scatter(phip/dtr, thetap/dtr, s=100, color='red', zorder=4, label='Ponto 2')
        ax.plot(np.linspace(phi1, phi2, 100)/dtr, np.linspace(theta1, theta2, 100)/dtr, dashes=[4, 4], zorder=3)
        ax.set_title('Geometria do patch')
        ax.grid(False)
        ax.set_xlim(phi1c/dtr - 5*h/(a*dtr), phi2c/dtr + 5*h/(a*dtr))
        ax.set_ylim(theta1c/dtr - 5*h/(a*dtr), theta2c/dtr + 5*h/(a*dtr))
        ax.set_facecolor('lightgray')
        plt.ylabel(r'$\theta$'+' (°)' )
        plt.xlabel(r'$\varphi$'+' (°)' )
        plt.gca().invert_yaxis()
        figures.append(fig)

    # Exportar resultado; dtc e dpc em graus
    parametros = f"Dtheta = {Dtheta}\nDphi = {Dphi}\nthetap = {thetap}\nphip = {phip}\np = {p}\nf = {flm_des}\ndtc = {0.0}\ndpc = {0.0}"
    out_dir = 'Resultados/Param'
    out_file = os.path.join(out_dir, 'CP_Param.txt')
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w') as f:
        f.write(parametros)
    if show:
        print(f"Arquivo salvo em {out_file}")

    for i, fig in enumerate(figures):
        fig.savefig(os.path.join(output_folder, f'figure_{i+1}.eps'), format='eps')
        fig.savefig(os.path.join(output_folder, f'figure_{i+1}.png'))
    
fim_total = time.time()
print("\n\nTempo total para o fim do código: ", fim_total - inicio_total, "segundos\n\n")

if show:
    plt.show()