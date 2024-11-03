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
L, M = 2, 2             # Modo em teste

# Constantes gerais
dtr = np.pi/180         # Graus para radianos (rad/°)
e0 = 8.854e-12          # (F/m)
u0 = np.pi*4e-7         # (H/m)
c = 1/np.sqrt(e0*u0)    # Velocidade da luz no vácuo (m/s)
gamma = 0.577216        # Constante de Euler-Mascheroni
eps = 1e-5              # Limite para o erro numérico

dtc = 0 * dtr
dpc = 0 * dtr
# Geometria da cavidade
a = 100e-3              # Raio da esfera de terra (m)
h = 1.524e-3            # Espessura do substrato dielétrico (m)
a_bar = a + h/2         # Raio médio da esfera de terra (m)
b = a + h               # Raio exterior do dielétrico (m)
theta1c = 66.73 * dtr + dtc     # Ângulo de elevação 1 da cavidade (rad)
theta2c = 113.27 * dtr + dtc    # Ângulo de elevação 2 da cavidade (rad)
phi1c = 72.4 * dtr + dpc        # Ângulo de azimutal 1 da cavidade (rad)
phi2c = 107.6 * dtr + dpc       # Ângulo de azimutal 2 da cavidade (rad)

deltatheta1 = h/a       # Largura angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = h/a       # Largura angular 2 do campo de franjas e da abertura polar (rad)
theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2 # Ângulos polares físicos do patch (rad)
deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)
phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi             # Ângulos azimutais físicos do patch (rad)

# Coordenadas dos alimentadores coaxiais (rad)
thetap = [90.0 * dtr, 81.0 * dtr] 
phip = [82.4  * dtr, 90.0 * dtr]
probes = len(thetap)    # Número de alimentadores
df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
er = 2.55               # Permissividade relativa do substrato dielétrico
es = e0 * er            # Permissividade do substrato dielétrico (F/m)
Dphip = [np.exp(1.5)*df/(2*a*np.sin(t)) for t in thetap]     # Comprimento angular do modelo da ponta de prova (rad)

# Outras variáveis
tgdel = 0.022           # Tangente de perdas
sigma = 5.8e50          # Condutividade elétrica dos condutores (S/m)
Z0 = 50                 # Impedância intrínseca (ohm)

output_folder = 'Resultados/Analise_Geral_TM'+str(L)+str(M)
os.makedirs(output_folder, exist_ok=True)
figures = []

# Funções associadas de Legendre para |z| < 1
def legP(z, l, m):
    u = m * np.pi / (phi2c - phi1c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if u.is_integer():
            return np.where(np.abs(z) < 1, (-1)**u * (sp.gamma(l+u+1)/sp.gamma(l-u+1)) * np.power((1-z)/(1+z),u/2) * sp.hyp2f1(-l,l+1,1+u,(1-z)/2)  / sp.gamma(u+1), 1)
        else:
            return np.where(np.abs(z) < 1, np.power((1+z)/(1-z),u/2) * sp.hyp2f1(-l,l+1,1-u,(1-z)/2)  / sp.gamma(1-u), 1)

def legQ(z, l, m):
    u = m * np.pi / (phi2c - phi1c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

def Equation(l, theta1f, theta2f, m):
    return np.where(np.abs(np.cos(theta1f)) < 1 and np.abs(np.cos(theta2f)) < 1, DP(theta1f,l,m)*DQ(theta2f,l,m)-DQ(theta1f,l,m)*DP(theta2f,l,m) , 1)

Lambda = np.linspace(-0.1, 32, 64) # Domínio de busca
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
        if r < n * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
            k += 1
    if show:
        print("m =", n, ":", [round(r, 5) for r in roots[k:k+5]], ', mu =', n * np.pi / (phi2c - phi1c), "\n\n")
    rootsLambda.append(roots[k:k+5])

if show:
    print('Raízes lambda:')
for n in range(0,5):
    root_find(n)

fim = time.time()
if show:
    print("Tempo decorrido:", fim - inicio_total, "segundos\n\n")

mz = 1
# Equação para cada m
fig = plt.figure()
plt.plot(Lambda, Equation(Lambda, theta1c, theta2c, mz), label='m=1')
plt.title('Equação da condição de contorno')
plt.ylabel(r'$\chi_{\mu}(\lambda)$')
plt.xlabel(r'$\lambda$')
plt.grid(True)
figures.append(fig)

# Raízes inválidas
itera = 0
Um = mz*np.pi/(phi2c-phi1c)
Lambda = np.linspace(0, Um, 84*mz)
Eq = Equation(Lambda, theta1c, theta2c, mz)
fig = plt.figure()
plt.plot(Lambda, Eq, label='m=1')
plt.axvline(x=Um, color='r', linestyle='--')
while Um - itera > 0:
    plt.plot(Um - itera, 0, 'o', color='r', markersize=5)
    itera += 1
plt.title('Raízes inválidas da equação de contorno')
plt.ylabel(r'$\chi_{\mu}(\lambda)$')
plt.xlabel(r'$\lambda$')
plt.grid(True)
figures.append(fig)

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
if show:
    print('Lambdas para TM01 e TM10: ', rootsLambda[0][1],' e ', rootsLambda[1][0])

def R(v,l,m):                                  # Função auxiliar para os campos elétricos
    Lamb = rootsLambda[l][m]
    return (DP(theta1c,Lamb,m)*legQ(v,Lamb,m) - DQ(theta1c,Lamb,m)*legP(v,Lamb,m)) * np.sin(theta1c)

def Er_lm_norm(theta,phi,l,m):                 # Campo elétrico dos modos 'normalizados' da solução homogênea
    u = m * np.pi / (phi2c - phi1c)
    return R(np.cos(theta),l,m) * np.cos(u * (phi - phi1c))


# Verificação da paridade de R10 e R01
fig = plt.figure()
plt.plot(np.arange(0,359,2), R(np.cos(np.arange(0,359,2) * dtr),1,0) + R(np.cos((180-np.arange(0,359,2))* dtr),1,0))
plt.title(r'$R^{0}_{\lambda_{10}}(x) + R^{0}_{\lambda_{10}}(-x) = 0$')
plt.ylabel(r'$R^{0}_{\lambda_{10}}(x) + R^{0}_{\lambda_{10}}(-x)$')
plt.xlabel(r'$x$')
plt.grid(True)
figures.append(fig)
    
fig = plt.figure()
plt.plot(np.linspace(-1, 1, 604), R(np.linspace(-1, 1, 604),1,0))
plt.title(r'$R^{0}_{\lambda_{10}}$'+': Função ímpar')
plt.ylabel(r'$R^{0}_{\lambda_{10}}(x)$')
plt.xlabel(r'$x$')
plt.grid(True)
figures.append(fig)
    
lim = 0.9
fig = plt.figure()
plt.plot(np.linspace(-lim, lim, 604), R(np.linspace(-lim, lim, 604),0,1))
plt.title(r'$R^{\mu_{01}}_{\lambda_{01}}$'+': Função par')
plt.ylabel(r'$R^{\mu_{01}}_{\lambda_{01}}(x)$')
plt.xlabel(r'$x$')
plt.grid(True)
figures.append(fig)

phi = np.linspace(phi1c, phi2c, 200)           # Domínio de phi (rad)
theta = np.linspace(theta1c, theta2c, 200)     # Domínio de theta (rad)

P, T = np.meshgrid(phi, theta)
Er_LM = Er_lm_norm(T, P, L, M)
Amp = np.abs(Er_LM/np.max(np.abs(Er_LM)))      # Normalizado
Phase = np.angle(Er_LM)

if show:
    print('\nMáximo do campo', np.max(Er_LM))
    print('Modo testado: TM', L, M)
    print('Frequência do modo testado: ', flm[L][M] / 1e9)

# Mapa de Amplitude
fig = plt.figure(figsize=(8, 6))
plt.contourf(P / dtr, T / dtr, Amp, cmap='jet', levels = 300)
plt.colorbar()
plt.xlabel(r'$\varphi$' + ' (graus)', fontsize=14)
plt.ylabel(r'$\theta$' + ' (graus)', fontsize=14)
plt.title('Mapa de Amplitude (Normalizado)')
figures.append(fig)

# Mapa de Fase
fig = plt.figure(figsize=(8, 6))
plt.contourf(P / dtr, T / dtr, Phase / dtr, cmap='gnuplot', levels = 200)
plt.colorbar(label = 'Fase (grau)')
plt.xlabel(r'$\varphi$' + ' (graus)', fontsize=14)
plt.ylabel(r'$\theta$' + ' (graus)', fontsize=14)
plt.title('Mapa de Fase')
figures.append(fig)

# Impedância de entrada - Circuito RLC:
def R2(v, l, m):                               # Quadrado da função auxiliar para os campos elétricos
    return R(v, l, m)**2

def tgef(L,M):
    Rs = np.sqrt(2 * np.pi * flm[L][M] * u0 / (2 * sigma))
    Qc = 2 * np.pi * flm[L][M] * u0 * h / (2 * Rs)
    Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)
    return tgdel + 1/Qc

def RLC(f, L, M, p1, p2):
    U = M * np.pi / (phi2c - phi1c)
    Ilm = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]

    # Auxiliar alpha
    alpha = h * (R(np.cos(thetap[p1-1]), L, M) * np.cos(U * (phip[p1-1]-phi1c)) * np.sinc(U * Dphip[p1-1] / (2*np.pi))) * \
        (R(np.cos(thetap[p2-1]), L, M) * np.cos(U * (phip[p2-1]-phi1c)) * np.sinc(U * Dphip[p2-1] / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
    if M != 0:
        alpha = 2 * alpha
    
    # Componentes RLC
    RLM = alpha/(2 * np.pi * f * tgef(L,M))
    CLM = 1/alpha
    LLM = alpha/(2 * np.pi * flm[L][M])**2
    return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM)))

def Z(f, L, M, p1, p2):
    k = (2 * np.pi * f) * np.sqrt(es * u0) 
    eta = np.sqrt(u0 / es)
    Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
    return RLC(f, L, M, p1, p2) + 1j*Xp

def Zlm(f, L, M):                              # Matriz impedância
    Zmatrix = []
    for q in range(probes):  
        line = []
        for p in range(probes):
            line.append(Z(f,L,M,q+1,p+1))
        Zmatrix.append(line)
    return Zmatrix
Zlm = np.vectorize(Zlm)

def Slm(f, L, M):                              # Matriz de espalhamento
    return (np.linalg.inv(np.array(Zlm(f, L,M))/Z0 + np.eye(probes)) * (np.array(Zlm(f, L,M))/Z0 - np.eye(probes))).tolist()
Slm = np.vectorize(Slm)

freqs01 = np.linspace(1.5, 1.7, 641) * 1e9
freqs10 = np.linspace(1.1, 1.2, 641) * 1e9

if show:
    print('\nMatriz Z TM'+str(L)+str(M)+': ', Zlm(flm[L][M],L,M))
    print('\nMatriz S TM'+str(L)+str(M)+': ', Slm(flm[L][M],L,M))

# Modo TM01 com a ponta de prova 1
if show:
    print('\nTM01: ', flm[0][1] / 1e9)
fig = plt.figure()
plt.plot(freqs01 / 1e9, np.real(Z(freqs01, 0, 1, 1, 1)), label='re(Z11)')
plt.plot(freqs01 / 1e9, np.imag(Z(freqs01, 0, 1, 1, 1)), label='im(Z11)')
plt.plot(freqs01 / 1e9, [Z0] * len(freqs01), label='Z0')
plt.axvline(x=flm[0][1] / 1e9, color='r', linestyle='--')
plt.title('Impedância TM01 - Ponta 1')
plt.xlabel('Frequência (GHz)')
plt.ylabel('Impedância ' + r'($\Omega$)')
plt.legend()
plt.grid(True)
figures.append(fig)

# Modo TM10 com a ponta de prova 2
if show:
    print('TM10: ', flm[1][0] / 1e9, '\n')
fig = plt.figure()
plt.plot(freqs10 / 1e9, np.real(Z(freqs10, 1, 0, 2, 2)), label='re(Z22)')
plt.plot(freqs10 / 1e9, np.imag(Z(freqs10, 1, 0, 2, 2)), label='im(Z22)')
plt.plot(freqs10 / 1e9, [Z0] * len(freqs10), label='Z0')
plt.axvline(x=flm[1][0] / 1e9, color='r', linestyle='--')
plt.title('Impedância TM10 - Ponta 2')
plt.xlabel('Frequência (GHz)')
plt.ylabel('Impedância ' + r'($\Omega$)')
plt.legend()
plt.grid(True)
figures.append(fig)

# Impedâncias de entrada, considerando Iin1 = Iin2
freqs = np.linspace(1.4, 1.7, 641) * 1e9
Iin = [1, 1]
Zin1 = Z(freqs,0,1,1,1) + Z(freqs,0,1,1,2) * Iin[1] / Iin[0]
Zin2 = Z(freqs,0,1,2,1) * Iin[0] / Iin[1] + Z(freqs,0,1,2,2)

# Impedância
fig = plt.figure()
plt.plot(freqs / 1e9, np.real(Zin1), label='re(Zin1)')
plt.plot(freqs / 1e9, np.imag(Zin1), label='im(Zin1)')
plt.plot(freqs / 1e9, [Z0] * len(freqs), label='Z0')
plt.title('Impedância de entrada na ponta 1 - TM01')
plt.xlabel('Frequência (GHz)')
plt.ylabel('Impedância ' + r'($\Omega$)')
plt.legend()
plt.grid(True)
figures.append(fig)

# Dados do HFSS
data_hfss = pd.read_csv('HFSS/Cavidade/Z Parameter Plot 1_compare.csv')
freqs_hfss = data_hfss['Freq [GHz]'] # GHz
re_hfss = data_hfss['re(Z(1,1)) []']
im_hfss = data_hfss['im(Z(1,1)) []']

# Carta de Smith
figSm = go.Figure()
Zchart = Z(freqs01, 0, 1, 1, 1)/Z0 # Multiplicar fora do argumento da função abaixo
figSm.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="red", name="Z(1,1)"))
Zchart = Z(freqs10, 1, 0, 2, 2)/Z0
figSm.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="green", name="Z(2,2)"))
Zchart =(re_hfss+1j*im_hfss)/Z0
figSm.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="blue", name=r"$Z_{HFSS}$"))
figSm.write_image(output_folder+"/Smith.png")

# Comparação de impedâncias
fig = plt.figure()
# Simulados
plt.plot(freqs_hfss, re_hfss, label='re(Z(1,1) simulado')
plt.plot(freqs_hfss, im_hfss, label='im(Z(1,1)) simulado')
plt.plot(freqs_hfss, [Z0] * len(freqs_hfss), label='Z0')
# Teóricos
plt.plot(freqs_hfss, np.real(Z(freqs_hfss * 1e9, 0, 1, 1, 1)), label='re(Z(1,1) modelo')
plt.plot(freqs_hfss, np.imag(Z(freqs_hfss * 1e9, 0, 1, 1, 1)), label='im(Z(1,1)) modelo')
plt.title('Impedância: Modelo x Simulação')
plt.xlabel('Frequência (GHz)')
plt.ylabel('Impedância ' + r'($\Omega$)')
plt.legend()
plt.grid(True)
figures.append(fig)

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

eps = 1e-30
l = m = 35 # m <= l
Ml, Mm = np.meshgrid(np.arange(0,l+1), np.arange(0,m+1))
delm = np.ones(m+1)
delm[0] = 0.5
flm_des = 1575.42e6 # Hz

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    S_lm = (2 * Ml * (Ml+1) * sp.gamma(1+Ml+Mm)) / ((2*Ml+1) * sp.gamma(1+Ml-Mm))
    S_lm += (1-np.abs(np.sign(S_lm)))*eps
    I_th = np.array([[spi.quad(partial(Itheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
    I_dth = np.array([[spi.quad(partial(IDtheta, L = i, M = j), theta1 , theta2)[0] for i in range(l+1)] for j in range(m+1)])
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
        Ethv = ((1j ** Ml) * (b * sp.lpmn(m, l, np.cos(theta))[1] * (Eth0_A * IdP_1 + Eth0_C * IdP_2) * (-np.sin(theta))/ (dH2_dr) \
            + 1j * Mm**2 * sp.lpmn(m, l, np.cos(theta))[0] * (Eth0_A * IpP_1 + Eth0_C * IpP_2) / (k * np.sin(theta) * H2) ) * \
            (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) * np.cos(Mm * ((phi1 + phi2)/2 - phi)) / (np.pi * S_lm))
        return np.sum(np.dot(delm, Ethv))
        
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
        Ephv = ((1j ** Ml) * (b * sp.lpmn(m, l, np.cos(theta))[0] * (Eth0_A * IdP_1 + Eth0_C * IdP_2)/ (np.sin(theta) * dH2_dr) \
            + 1j * sp.lpmn(m, l, np.cos(theta))[1] * (Eth0_A * IpP_1 + Eth0_C * IpP_2) * (-np.sin(theta)) / (k * H2) ) * \
            2 * np.sin(Mm * (phi2-phi1)/2) * np.sin(Mm * ((phi1 + phi2)/2 - phi)) / (np.pi * S_lm))
        return np.sum(np.dot(delm, Ephv))

def Eth_h_prot(theta, phi):
    k = 2 * np.pi * flm_des / c
    Dphic = phi1 - phi1c
    Eph0 = 1

    dH2_dr = np.tile(schelkunoff2(l, k * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, k * b),(m + 1, 1))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

inicio = time.time()
angulos = np.arange(0,360,1) * dtr + eps

fig, axs = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': 'polar'})

# Plotar o primeiro gráfico
axs[0, 0].plot(angulos, E_v((theta1c + theta2c) / 2 + eps, angulos))
axs[0, 0].set_title('Amplitude do Campo Elétrico TM01 (dB)')
axs[0, 0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0, 0].grid(True)
axs[0, 0].set_theta_zero_location('N')
axs[0, 0].set_theta_direction(-1)
axs[0, 0].set_rlim(-30,0)

# Plotar o segundo gráfico
axs[0, 1].plot(angulos[::1], E_h((theta1c + theta2c) / 2 + eps, angulos))
axs[0, 1].set_title('Amplitude do Campo Elétrico TM01 (dB)')
axs[0, 1].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0, 1].grid(True)
axs[0, 1].set_theta_zero_location('N')
axs[0, 1].set_theta_direction(-1)
axs[0, 1].set_rlim(-30,0)

# Plotar o terceiro gráfico
axs[1, 0].plot(angulos[::1], E_v(angulos, (phi1c + phi2c) / 2 + eps), color='red')
axs[1, 0].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[1, 0].grid(True)
axs[1, 0].set_theta_zero_location('N')
axs[1, 0].set_theta_direction(-1)
axs[1, 0].set_rlim(-30,0)

# Plotar o quarto gráfico
axs[1, 1].plot(angulos[::1], E_h(angulos, (phi1c + phi2c) / 2 + eps), color='red')
axs[1, 1].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[1, 1].grid(True)
axs[1, 1].set_theta_zero_location('N')
axs[1, 1].set_theta_direction(-1)
axs[1, 1].set_rlim(-30,0)
figures.append(fig)
plt.tight_layout()

fim = time.time()
if show:
    print("Tempo decorrido para o cálculo dos 4 campos e gráficos: ", fim - inicio, "segundos\n\n")

for i, fig in enumerate(figures):
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.eps'), format='eps')
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.png'))
fim_total = time.time()
print("Tempo total para o fim do código: ", fim_total - inicio_total, "segundos\n\n")

if show:
    plt.show()
    figSm.show()