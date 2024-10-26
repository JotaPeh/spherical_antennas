from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
import warnings
import plotly.graph_objects as go
import pandas as pd
import os

inicio_total = time.time()
show = 0
flm_des = 1575.42e6     # Frequência de operação desejada
Ltot = Mtot = 7         # Modo TM_LM em teste, TM10 ou TM01 apenas

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

theta1c = 72.87033867178677 * dtr + dtc      # Ângulo de elevação 1 da cavidade (rad)
theta2c = 107.12966132821323 * dtr + dtc     # Ângulo de elevação 2 da cavidade (rad)
phi1c = 67.73144027332279 * dtr + dpc        # Ângulo de azimutal 1 da cavidade (rad)
phi2c = 112.26855972667721 * dtr + dpc       # Ângulo de azimutal 2 da cavidade (rad)
thetap = [96.68711790508867 * dtr]  
phip = [90.0 * dtr]

# if M == 0:
#     theta1c = 72.87033867178677 * dtr + dtc      # Ângulo de elevação 1 da cavidade (rad)
#     theta2c = 107.12966132821323 * dtr + dtc     # Ângulo de elevação 2 da cavidade (rad)
#     phi1c = 67.73144027332279 * dtr + dpc        # Ângulo de azimutal 1 da cavidade (rad)
#     phi2c = 112.26855972667721 * dtr + dpc       # Ângulo de azimutal 2 da cavidade (rad)
# elif M == 1:
#     theta1c = 57.3524873620928 * dtr + dtc       # Ângulo de elevação 1 da cavidade (rad)
#     theta2c = 162.6475126379072 * dtr + dtc      # Ângulo de elevação 2 da cavidade (rad)
#     phi1c = 62.57883643237908 * dtr + dpc        # Ângulo de azimutal 1 da cavidade (rad)
#     phi2c = 97.42116356762092 * dtr + dpc       # Ângulo de azimutal 2 da cavidade (rad)

Dtheta = theta2c - theta1c
Dphi = phi2c - phi1c
deltatheta1 = h/a       # Largura angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = h/a       # Largura angular 2 do campo de franjas e da abertura polar (rad)
theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2 # Ângulos polares físicos do patch (rad)
deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)
phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi             # Ângulos azimutais físicos do patch (rad)

# Coordenadas dos alimentadores coaxiais (rad)
# if M == 0:
#     thetap = [96.68711790508867 * dtr]  
#     phip = [90.0 * dtr]
# elif M == 1:
#     thetap = [90.0 * dtr]  
#     phip = [96.75123556473255 * dtr]
probes = len(thetap)    # Número de alimentadores
df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
er = 2.55               # Permissividade relativa do substrato dielétrico
es = e0 * er            # Permissividade do substrato dielétrico (F/m)
Dphip = [np.exp(1.5)*df/(2*a*np.sin(t)) for t in thetap]     # Comprimento angular do modelo da ponta de prova (rad)

# Outras variáveis
tgdel = 0.0022          # Tangente de perdas
sigma = 5.8e50          # Condutividade elétrica dos condutores (S/m)
Z0 = 50                 # Impedância intrínseca (ohm)

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

# Início
def Equation(l, theta1f, theta2f, m):
    return np.where(np.abs(np.cos(theta1f)) < 1 and np.abs(np.cos(theta2f)) < 1, DP(theta1f,l,m)*DQ(theta2f,l,m)-DQ(theta1f,l,m)*DP(theta2f,l,m) , 1)

Lambda = np.linspace(-0.1, 132, 164) # Domínio de busca, o professor usou a janela de 0.1
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
    print("n =", n, ":", [round(r, 5) for r in roots[k:k+Ltot+1]], ', mu =', n * np.pi / (phi2c - phi1c), "\n\n")
    rootsLambda.append(roots[k:k+Ltot+1])

if show:
    print('Raízes lambda:')
for n in range(0,Ltot+1):
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

spfac = []
erros = []
for L in range(Ltot):
    for M in range(Mtot):

        def R(v,l,m):                                  # Função auxiliar para os campos elétricos
            Lamb = rootsLambda[L][M]
            # return (DP(theta1c,Lamb,m)*legQ(v,Lamb,m) - DQ(theta1c,Lamb,m)*legP(v,Lamb,m)) * np.sin(theta1c)
            # print(DP(theta1c,Lamb,m))
            # return (legQ(v,l,m)-legP(v,l,m)) * np.sin(theta1c)
            return (DP(theta1c,Lamb,m)*legQ(v,l,m) - DQ(theta1c,Lamb,m)*legP(v,l,m)) * np.sin(theta1c)

        def R2(v, l, m):                               # Quadrado da função auxiliar para os campos elétricos
            return R(v, l, m)**2

        def PQ(v, m):                               # Quadrado da função auxiliar para os campos elétricos
            return legP(v,rootsLambda[L][M],m)*legQ(v,rootsLambda[L+1][M],m)


        print('Lambda/Mu: ', rootsLambda[L][M], '/',  M * np.pi / (phi2c - phi1c))
        Ilm = spi.quad(partial(PQ, m = M),np.cos(theta2c),np.cos(theta1c))[0]

        print('Ambas: ', Ilm)

        v = rootsLambda[L][M]

        temponumIn = time.time()
        Ilm = spi.quad(partial(R2, l = v, m = M),np.cos(theta2c),np.cos(theta1c))[0]
        temponumFim = time.time()

        print('Modo: ', L, M)

        print('Numérico:  ', Ilm)

        u = M * np.pi / (phi2c - phi1c)
        v = rootsLambda[L][M]
        delta = 5e-6
        def Integ(x, l, m):
            return (-x * (R(x,v,m))**2 + R(x,v+1,m)*R(x,v,m)+(v-u+1)*(R(x,v,m)*(R(x,v+1+delta,m)-R(x,v+1-delta,m))-R(x,v+1,m)*(R(x,v+delta,m)-R(x,v-delta,m)))/(2*delta))/ (2*v+1)

        tempoanIn = time.time()
        valueAn = Integ(np.cos(theta1c),L,M)-Integ(np.cos(theta2c),L,M)
        tempoanFim = time.time()

        print('Analítico: ', valueAn)

        fact = (tempoanFim-tempoanIn)/(temponumFim-temponumIn)
        spfac.append(fact)
        print('Speed-up factor: ', fact)
        
        erros.append(np.abs(valueAn-Ilm)/Ilm)
        print('Erro (%): ', np.abs(valueAn-Ilm)/Ilm, '\n\n')

        # Rodar alguns casos comparando tempo e formando o mesmo gráfico do artigo

print(erros, '\n', np.max(erros), '\n\n')
print(spfac, '\n', np.max(spfac))