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
import os
from sympy import symbols, Eq, solve

inicio_total = time.time()
show = 0
test = 0
project = 0

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
flm_des = 1575.42e6 # Banda L1

# Projeto Linear
klm = 2*np.pi*flm_des*np.sqrt(u0*es)
Dtheta = Theta_find(klm)[0]
Dphi = Phi_find(klm, Dtheta)[-1]

Dphia = Dphi - 2*deltaPhi
Dthetaa = Dtheta - 2*deltatheta1
Dphit = Dthetat = 0

# Primeira Iteração: Resultados da antena linear
Dphia *= 1.613e9/flm_des
Dthetaa *= 1.613e9/flm_des
Band = 1588.5-1565.1 # MHz
phip, thetap = 90, 95.2012 # thetap = 95.2012°

theta1, theta2 = np.pi/2 - Dthetaa/2, np.pi/2 + Dthetaa/2
phi1, phi2 = np.pi/2 - Dphia/2, np.pi/2 + Dphia/2

print('Dphia, Dthetaa: ', Dphia/dtr, Dthetaa/dtr)

def Geotruc(alphac, off = 0):
    P2 = np.array([b*np.cos(phi2)*np.sin(theta1), b*np.sin(phi2)*np.sin(theta1), b*np.cos(theta1)])
    
    alphay = np.arctan(1/(np.tan(theta1)*np.cos(phi1)))

    alphaa = np.arccos(-np.sin(theta1)**2*np.cos(2*phi1)-np.cos(theta1)**2)
    alphat = alphaa-alphac

    if not off:
        print('alphay, alphaa, alphat: ', alphay/dtr, alphaa/dtr, alphat/dtr)

    # Interseção com a lateral: 
    ul = -b/np.sqrt(1+np.tan((np.pi+alphat)/2)**2+((np.tan((np.pi+alphat)/2)/np.tan(phi2)-np.cos(alphay))/np.sin(alphay))**2)
    vl = ul*(np.tan((np.pi+alphat)/2)/np.tan(phi2)-np.cos(alphay))/np.sin(alphay)

    Dthetat = np.arccos((P2[0]*(ul*np.cos(alphay)+vl*np.sin(alphay)) + P2[1]*(ul*np.tan((np.pi+alphat)/2)) + P2[2]*(vl*np.cos(alphay)-ul*np.sin(alphay)))/b**2)

    if not off:
        print('Dthetat: ', Dthetat/dtr, alphac*np.sin(alphay)/dtr)

    # Interseção com lado superior:
    ur = symbols('ur')

    equation = Eq(ur**2*(1+np.tan((np.pi+alphat)/2)**2)+(b*np.cos(theta1)/np.cos(alphay)+ur*np.tan(alphay))**2-b**2, 0)

    solutions = solve(equation, ur)
    ur = float(np.min(solutions))
    vr = b*np.cos(theta1)/np.cos(alphay)+ur*np.tan(alphay)

    Dphit = np.arccos((P2[0]*(ur*np.cos(alphay)+vr*np.sin(alphay)) + P2[1]*(ur*np.tan((np.pi+alphat)/2)))/(b*np.sin(theta1))**2)

    if not off:
        print('Dphit: ', Dphit/dtr, alphac*np.cos(alphay)/np.sin(theta1)/dtr)
        return b**2 * Dphit * np.cos((Dthetaa-Dthetat)/2) * np.sin(Dphit/2), Dphit, Dthetat

    # Área:
    return b**2 * Dphit * np.cos((Dthetaa-Dthetat)/2) * np.sin(Dphit/2)

Atot = 2 * b**2 * Dphia *np.sin(Dthetaa/2)
print('Área total (mm²): ', Atot*1e6)

# Projeto de truncamento
def Trunc(alphac):
    return Geotruc(alphac, 1) - Atot*Band*1e6/(4*flm_des)

inicio_projeto = time.time()
root = root_scalar(Trunc, bracket=[1e-6, 20*dtr], method='bisect')
fim_projeto = time.time()
if show:
    print("\nTempo de projeto: ", fim_projeto-inicio_projeto,' s')

alphac = np.array(root.root)

print('\nalphac: ', alphac/dtr)
print('alphac/2 = ', alphac/2/dtr)

print('\nÁrea do chanfro (mm²): ', Geotruc(alphac, 1)*1e6, Atot*Band*1e12/(4*flm_des), '\n')
Armv, Dphit, Dthetat = Geotruc(alphac)
print('Área removida (2 chanfros, em mm²): ', Armv*2e6)


# Criação da figura e dos eixos
vertices = np.array([
    [phi1/dtr, theta1/dtr],
    [(phi2 - Dphit)/dtr, theta1/dtr],
    [phi2/dtr, (theta1 + Dthetat)/dtr],
    [phi2/dtr, theta2/dtr],
    [(phi1 + Dphit)/dtr, theta2/dtr],
    [phi1/dtr, (theta2 - Dthetat)/dtr]
])

fig, ax = plt.subplots()
polygon = patches.Polygon(vertices, closed=True, color='#B87333', zorder=2)
ax.add_patch(polygon)
ax.scatter(phip, thetap, s=50, color='orange', zorder=5, label='Ponto 1')  # Ponto 1
ax.scatter(phip, thetap, s=100, color='red', zorder=4, label='Ponto 2')    # Ponto 2

ax.plot(np.array([np.pi/2] * 100)/dtr, np.linspace(theta1, theta2, 100)/dtr, dashes=[4, 4], zorder=3)

ax.set_title('Geometria do Patch com Hexágono')
ax.set_xlim(phi1/dtr - 5, phi2/dtr + 5)
ax.set_ylim(theta1/dtr - 5, theta2/dtr + 5)
ax.set_facecolor('lightgray')

ax.set_title('Geometria do patch')
ax.grid(False)
ax.set_xlim(phi1/dtr - 5*h/(a*dtr), phi2/dtr + 5*h/(a*dtr))
ax.set_ylim(theta1/dtr - 5*h/(a*dtr), theta2/dtr + 5*h/(a*dtr))
ax.set_facecolor('lightgray')

theta1 -= deltatheta1
theta2 += deltatheta2
phi1 -= deltaPhi
phi2 += deltaPhi
vertices2 = np.array([
    [phi1/dtr, theta1/dtr],
    [(phi2 - Dphit)/dtr, theta1/dtr],
    [phi2/dtr, (theta1 + Dthetat)/dtr],
    [phi2/dtr, theta2/dtr],
    [(phi1 + Dphit)/dtr, theta2/dtr],
    [phi1/dtr, (theta2 - Dthetat)/dtr]
])

polygon2 = patches.Polygon(vertices2, closed=True, color='gray', zorder=1)
ax.add_patch(polygon2)

plt.ylabel(r'$\theta$'+' (°)' )
plt.xlabel(r'$\varphi$'+' (°)' )
plt.gca().invert_yaxis()

fig.savefig(os.path.join(output_folder, 'figure_Trunc.eps'), format='eps')
fig.savefig(os.path.join(output_folder, 'figure_Trunc.png'))
fim_total = time.time()
if show:
    print("Tempo total para o fim do código: ", fim_total - inicio_total, "segundos\n")

if show:
    plt.show()