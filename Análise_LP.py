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

filein = input("Digite o nome do arquivo (ex: 'LP_10_Param'): ")
inicio_total = time.time()
show = 0

parametros = {line.split(' = ')[0]: float(line.split(' = ')[1]) for line in open('Resultados/Param/' + filein + '.txt')}
L, M, Dtheta, Dphi, thetap, phip, flm_des = [parametros[key] for key in ['L', 'M', 'Dtheta', 'Dphi', 'thetap', 'phip', 'f']]
L = int(L)
M = int(M)

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
theta1c = np.pi/2 - Dtheta/2 + dtc      # Ângulo de elevação 1 da cavidade (rad)
theta2c = np.pi/2 + Dtheta/2 + dtc      # Ângulo de elevação 2 da cavidade (rad)
phi1c = np.pi/2 - Dphi/2 + dpc          # Ângulo de azimutal 1 da cavidade (rad)
phi2c = np.pi/2 + Dphi/2 + dpc          # Ângulo de azimutal 2 da cavidade (rad)

deltatheta1 = h/a       # Largura angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = h/a       # Largura angular 2 do campo de franjas e da abertura polar (rad)
theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2 # Ângulos polares físicos do patch (rad)
deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)
phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi             # Ângulos azimutais físicos do patch (rad)

# Coordenadas dos alimentadores coaxiais (rad)
thetap = [thetap]  
phip = [phip]
probes = len(thetap)    # Número de alimentadores
df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
er = 2.55               # Permissividade relativa do substrato dielétrico
es = e0 * er            # Permissividade do substrato dielétrico (F/m)
Dphip = [np.exp(1.5)*df/(2*a*np.sin(t)) for t in thetap]     # Comprimento angular do modelo da ponta de prova (rad)

# Outras variáveis
tgdel = 0.0022          # Tangente de perdas
sigma = 5.8e50          # Condutividade elétrica dos condutores (S/m)
Z0 = 50                 # Impedância intrínseca (ohm)

# path = 'HFSS/LP_pre/'   # Resultados anteriores à correção das franjas
path = 'HFSS/LP/'       # Resultados
output_folder = 'Resultados/Analise_LP_TM'+str(L)+str(M)
os.makedirs(output_folder, exist_ok=True)
figures = []

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
        if r < n * np.pi / (phi2c - phi1c) - 1 + eps and round(r, 5) != 0:
            k += 1
    if show:
        print("n =", n, ":", [round(r, 5) for r in roots[k:k+5]], ', mu =', n * np.pi / (phi2c - phi1c), "\n\n")
    rootsLambda.append(roots[k:k+5])

if show:
    print('Raízes lambda:')
for n in range(0,5):
    root_find(n)

fim = time.time()
if show:
    print("Tempo decorrido:", fim - inicio_total, "segundos\n\n")

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

phi = np.linspace(phi1c, phi2c, 200)           # Domínio de phi (rad)
theta = np.linspace(theta1c, theta2c, 200)     # Domínio de theta (rad)

kLM = 2*np.pi*flm[L][M]*np.sqrt(u0*es)

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

def tgefTot(klm, L, M):
    Rs = np.sqrt(klm * np.sqrt(u0/es) / (2 * sigma))
    Qc = klm * np.sqrt(u0/es) * h / (2 * Rs)
    Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)
    
    if M == 0:
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        K0 = klm / np.sqrt(er)

        I10 = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]

        IdP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[1] * np.sin((theta1+theta1c)/2)**2 * -deltatheta1
        IdP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[1] * np.sin((theta2+theta2c)/2)**2 * -deltatheta2
        IpP_1 = sp.lpmn(m, l, np.cos((theta1+theta1c)/2))[0] * deltatheta1
        IpP_2 = sp.lpmn(m, l, np.cos((theta2+theta2c)/2))[0] * deltatheta2
        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S10 =(((phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi))) ** 2 / (4*S_lm)) * (np.abs(b*(IdP_1+IdP_2)/dH2_dr)**2 + Mm**2 * np.abs((IpP_1+IpP_2)/(K0*H2))**2)
        S10 =  np.sum(np.dot(delm, S10))

        Q10 = (np.pi / 12) * klm * np.sqrt(er) * (b**3 - a**3) * I10 * Dphi / (np.abs(R(np.cos(theta1c),L,M))**2 * S10)
        return tgdel + 1/Qc + 1/Q10
    elif M == 1:
        theta1c, theta2c = np.pi/2 - Dtheta/2, np.pi/2 + Dtheta/2
        phi1c, phi2c = np.pi/2 - Dphi/2, np.pi/2 + Dphi/2
        theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2
        phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi
        K0 = klm / np.sqrt(er)

        I01 = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]

        dH2_dr = np.tile(schelkunoff2(l, K0 * b),(m + 1, 1))
        H2 = np.tile(hankel_spher2(l, K0 * b),(m + 1, 1))

        S01 = ((deltaPhi * np.sinc(Mm*deltaPhi/(2*np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2)) ** 2 / (4*S_lm)) * (Mm**2 * np.abs(b*I_th/dH2_dr)**2 + np.abs(I_dth/(K0*H2))**2)
        S01 =  np.sum(np.dot(delm, S01))

        Q01 = (np.pi / 96) * klm * np.sqrt(er) * (b**3 - a**3) * I01 * Dphi / (np.abs(R(np.cos((theta1c+theta2c)/2),L,M))**2 * S01)
        return tgdel + 1/Qc + 1/Q01
    
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

def Z(f, L, M, p1, p2):
    k = (2 * np.pi * f) * np.sqrt(es * u0) 
    eta = np.sqrt(u0 / es)
    Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
    U = M * np.pi / (phi2c - phi1c)
    Ilm = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))[0]
    Rs = np.sqrt(2 * np.pi * flm[L][M] * u0 / (2 * sigma))
    Qc = 2 * np.pi * flm[L][M] * u0 * h / (2 * Rs)
    Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)

    # Auxiliar alpha
    alpha = h * (R(np.cos(thetap[p1-1]), L, M) * np.cos(U * (phip[p1-1]-phi1c)) * np.sinc(U * Dphip[p1-1] / (2*np.pi))) * \
        (R(np.cos(thetap[p2-1]), L, M) * np.cos(U * (phip[p2-1]-phi1c)) * np.sinc(U * Dphip[p2-1] / (2*np.pi))) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
    if M != 0:
        alpha = 2 * alpha
    
    # Componentes RLC
    RLM = alpha/(2 * np.pi * f * (0.015 + 1/Qc))
    CLM = 1/alpha
    LLM = alpha/(2 * np.pi * flm[L][M])**2
    return 1/(1/RLM+1j*(2 * np.pi * f * CLM - 1/(2 * np.pi * f * LLM))) + 1j*Xp

def ZLP(f, p1, p2):
    k = (2 * np.pi * f) * np.sqrt(es * u0) 
    eta = np.sqrt(u0 / es)
    Xp = (eta * k * h / (2 * np.pi)) * (np.log(4 / (k * df)) - gamma)
    return RLC(f, kLM, L, M, p1, p2) + 1j*Xp

def Zlm(f):                              # Matriz impedância
    Zmatrix = []
    for q in range(probes):  
        line = []
        for p in range(probes):
            line.append(ZLP(f, q+1,p+1))
        Zmatrix.append(line)
    return Zmatrix

def Slm(f):                              # Matriz de espalhamento
    return (np.linalg.inv(np.array(Zlm(f))/Z0 + np.eye(probes)) * (np.array(Zlm(f))/Z0 - np.eye(probes))).tolist()
Slm = np.vectorize(Slm)

if show:
    print('\nMatriz Z TM01: ', Zlm(flm[L][M]))
    print('\nMatriz S TM01: ', Slm(flm[L][M]))

# Impedâncias de entrada, considerando Iin1 = Iin2
freqs = np.linspace(1.45, 1.75, 641) * 1e9

ZinP = ZLP(freqs, 1, 1)

fig = plt.figure()
plt.plot(freqs/1e9, np.real(ZinP), label='Re(Z)')
plt.plot(freqs/1e9, np.imag(ZinP), label='Im(Z)')
plt.plot(freqs/1e9, [Z0] * len(freqs), label=r'$Z_0=50\Omega$')
plt.axvline(x=flm_des / 1e9, color='r', linestyle='--')
plt.xlabel('Frequência (GHz)')
plt.ylabel('Impedância (' + r'$\Omega$' + ')')
plt.title('Impedância: Modelo')
plt.legend()
plt.grid(True)
figures.append(fig)

# Dados do HFSS
if path == 'HFSS/LP_pre/':
    if M == 0:
        data_hfss = pd.read_csv(path+'ZLP10 Parameter.csv')
    elif M == 1:
        data_hfss = pd.read_csv(path+'ZLP01 Parameter.csv')
elif path == 'HFSS/LP/':
    if M == 0:
        data_hfss = pd.read_csv(path+'Z Parameter Plot LP10.csv')
    elif M == 1:
        data_hfss = pd.read_csv(path+'Z Parameter Plot LP01.csv')
freqs_hfss = data_hfss['Freq [GHz]'] # GHz
re_hfss = data_hfss['re(Z(1,1)) []']
im_hfss = data_hfss['im(Z(1,1)) []']
Zin_hfss = re_hfss+1j*im_hfss

# Carta de Smith
figSm = go.Figure()
Zchart = ZinP/Z0 # Multiplicar fora do argumento da função abaixo
figSm.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="green", name="Zin modelo"))
Zchart = (re_hfss+1j*im_hfss)/Z0
figSm.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="red", name="Zin simulado"))
if show:
    print('\nFrequência de ressonância: ', freqs_hfss[np.argmax(re_hfss)], '\n')
figSm.write_image(output_folder+"/Smith.png")

fig = plt.figure()
# Simulados
plt.plot(freqs_hfss, re_hfss, label='Re(Z(1,1)) simulado', color = 'r')
plt.plot(freqs_hfss, im_hfss, label='Im(Z(1,1)) simulado', color = '#8B0000')
plt.plot(freqs_hfss, [Z0] * len(freqs_hfss), label='Z0', color = 'y')
if path == 'HFSS/LP_pre/':
    if M == 0:
        plt.axvline(x=1.605, color='b', linestyle='--')
    elif M == 1:
        plt.axvline(x=1.603, color='b', linestyle='--')
# Modelo analítico
plt.plot(freqs_hfss, np.real(ZLP(freqs_hfss*1e9, 1, 1)), label='Re(Z(1,1)) modelo', color = 'g')
plt.plot(freqs_hfss, np.imag(ZLP(freqs_hfss*1e9, 1, 1)), label='Im(Z(1,1)) modelo', color = '#006400')
plt.axvline(x=flm_des / 1e9, color='b', linestyle='--')
plt.title('Impedância: Modelo x Simulação')
plt.xlabel('Frequência (GHz)')
plt.ylabel(r'Impedância ($\Omega$)')
plt.legend()
plt.grid(True)
figures.append(fig)

fig = plt.figure()
plt.plot(freqs_hfss, 20*np.log10(np.abs((Zin_hfss-Z0)/(Zin_hfss+Z0))), label=r'$s_{11}$' + ' simulado')
if path == 'HFSS/LP_pre/':
    plt.plot(freqs/1e9, 20*np.log10(np.abs((ZinP-Z0)/(ZinP+Z0))), label=r'$s_{11}$' + ' modelo')
    if M == 0:
        plt.axvline(x=1.605, color='b', linestyle='--')
    elif M == 1:
        plt.axvline(x=1.603, color='b', linestyle='--')
elif path == 'HFSS/LP/':
    plt.plot(freqs/1e9, 20*np.log10(np.abs((ZinP-Z0)/(ZinP+Z0))), label=r'$s_{11}$' + ' modelo', linestyle='--')
plt.axhline(y=-10, color='r', linestyle='--')
plt.axvline(x=flm_des / 1e9, color='b', linestyle='--')
plt.xlabel('Frequência (GHz)')
plt.ylabel(r'$|\Gamma_{in}|$' + ' (dB)')
plt.title('Coeficiente de reflexão (' + r'$s_{11}$' + ')')
plt.legend()
plt.grid(True)
figures.append(fig)

# Campos distantes
eps = 1e-30

def Eth_v_prot(theta, phi):
    Eth0_A, Eth0_C = 1, 1
    k = 2 * np.pi * flm_des / c
    
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
    Eth0_A, Eth0_C = 1, 1
    k = 2 * np.pi * flm_des / c

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
    Eph0 = 1
    k = 2 * np.pi * flm_des / c
    Dphic = phi1 - phi1c

    dH2_dr = np.tile(schelkunoff2(l, k * b),(m + 1, 1))
    H2 = np.tile(hankel_spher2(l, k * b),(m + 1, 1))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ethh = ((-1j ** Ml) * (b * I_th * sp.lpmn(m, l, np.cos(theta))[1] * (-np.sin(theta)) / dH2_dr \
            + 1j * sp.lpmn(m, l, np.cos(theta))[0] * I_dth / (k * H2 * np.sin(theta)) ) * \
            4 * Eph0 * np.sin(Mm*Dphic/2) * np.cos(Mm*(phi2-phi1+Dphic)/2) * np.sin(Mm * ((phi1 + phi2)/2 - phi)) / (np.pi * S_lm))
        return np.sum(np.dot(delm, Ethh))

def Eph_h_prot(theta, phi):
    Eph0 = 1
    k = 2 * np.pi * flm_des / c
    Dphic = phi1 - phi1c

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

if show:
    inicio = time.time()
    testEv = E_v(90 * dtr+eps, np.arange(0,360,1) * dtr+eps)
    testEh = E_h(90 * dtr+eps, np.arange(0,360,1) * dtr+eps)
    fim = time.time()
    print("Tempo decorrido para o cálculo paralelo: ", fim - inicio, "segundos\n\n")

# Gráficos polares
inicio = time.time()
angulos = np.arange(0,360,1) * dtr + eps
angles = np.arange(0, 360, 30)
angles_labels = ['0°', '30°', '60°', '90°', '120°', '150°', '180°', '-150°', '-120°', '-90°', '-60°', '-30°']

fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'})

if M == 0:
    if path == 'HFSS/LP_pre/':
        dataTh = pd.read_csv(path+'Ant10_Theta90.csv')
        angTh = dataTh['Phi [deg]'] * dtr + eps
        gainTh = dataTh['dB(GainTheta) [] - Freq=\'1.608GHz\' Theta=\'90.0000000000002deg\'']
        GainLin = np.max(gainTh)
        gainTh -= np.max(gainTh)
        dataPh = pd.read_csv(path+'Ant10_Phi90.csv')
        angPh = dataPh['Theta [deg]'] * dtr + eps
        gainPh = dataPh['dB(GainTheta) [] - Freq=\'1.608GHz\' Phi=\'90.0000000000002deg\'']
        gainPh -= np.max(gainPh)
    elif path == 'HFSS/LP/':
        dataTh = pd.read_csv(path+'Gain Plot in theta LP10.csv')
        angTh = dataTh['Phi [deg]'] * dtr + eps
        gainTh = dataTh['dB(GainTheta) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
        GainLin = np.max(gainTh)
        gainTh -= np.max(gainTh)
        dataPh = pd.read_csv(path+'Gain Plot in phi LP10.csv')
        angPh = dataPh['Theta [deg]'] * dtr + eps
        gainPh = dataPh['dB(GainTheta) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']
        gainPh -= np.max(gainPh)

    # Plotar o primeiro gráfico
    axs[0].plot(angulos, E_v((theta1c + theta2c) / 2 + eps, angulos), label = 'Modelo')
    axs[0].plot(angTh, gainTh, color='red', label = 'Simulação')
    axs[0].set_title('Plano H')
    axs[0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
    axs[0].grid(True)
    axs[0].set_theta_zero_location('N')
    axs[0].set_theta_direction(-1)
    axs[0].set_rlim(-30,0)
    axs[0].set_thetagrids(angles)
    axs[0].set_rlabel_position(45)

    axs[1].plot(angulos, E_v(angulos, (phi1c + phi2c) / 2 + eps), label = 'Modelo')
    axs[1].plot(angPh, gainPh, color='red', label = 'Simulação')
    axs[1].set_title('Plano E')
    axs[1].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
    axs[1].grid(True)
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)
    axs[1].set_rlim(-30,0)
    axs[1].set_thetagrids(angles, labels=angles_labels)
    axs[1].set_rlabel_position(45)

    fig.suptitle('Amplitude do Campo Elétrico TM10 (dB)')
    handles, figlabels = axs[0].get_legend_handles_labels()
    fig.legend(handles, figlabels, loc='lower center', ncol=1)


elif M == 1:
    if path == 'HFSS/LP_pre/':
        dataTh = pd.read_csv(path+'Ant01_Theta90.csv')
        angTh = dataTh['Phi [deg]'] * dtr + eps
        gainTh = dataTh['dB(GainPhi) [] - Freq=\'1.603GHz\' Theta=\'90deg\'']
        GainLin = np.max(gainTh)
        gainTh -= np.max(gainTh)
        dataPh = pd.read_csv(path+'Ant01_Phi90.csv')
        angPh = dataPh['Theta [deg]'] * dtr + eps
        gainPh = dataPh['dB(GainPhi) [] - Freq=\'1.603GHz\' Phi=\'90deg\'']
        gainPh -= np.max(gainPh)
    elif path == 'HFSS/LP/':
        dataTh = pd.read_csv(path+'Gain Plot in theta LP01.csv')
        angTh = dataTh['Phi [deg]'] * dtr + eps
        gainTh = dataTh['dB(GainPhi) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
        GainLin = np.max(gainTh)
        gainTh -= np.max(gainTh)
        dataPh = pd.read_csv(path+'Gain Plot in phi LP01.csv')
        angPh = dataPh['Theta [deg]'] * dtr + eps
        gainPh = dataPh['dB(GainPhi) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']
        gainPh -= np.max(gainPh)

    axs[0].plot(angulos, E_h((theta1c + theta2c) / 2 + eps, angulos), label = 'Modelo')
    axs[0].plot(angTh, gainTh, color='red', label = 'Simulação')
    axs[0].set_title('Plano E')
    axs[0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
    axs[0].grid(True)
    axs[0].set_theta_zero_location('N')
    axs[0].set_theta_direction(-1)
    axs[0].set_rlim(-30,0)
    axs[0].set_thetagrids(angles)
    axs[0].set_rlabel_position(45)

    axs[1].plot(angulos, E_h(angulos, (phi1c + phi2c) / 2 + eps), label = 'Modelo')
    axs[1].plot(angPh, gainPh, color='red', label = 'Simulação')
    axs[1].set_title('Plano H')
    axs[1].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
    axs[1].grid(True)
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)
    axs[1].set_rlim(-30,0)
    axs[1].set_thetagrids(angles, labels=angles_labels)
    axs[1].set_rlabel_position(45)

    fig.suptitle('Amplitude do Campo Elétrico TM01 (dB)')
    handles, figlabels = axs[0].get_legend_handles_labels()
    fig.legend(handles, figlabels, loc='lower center', ncol=1)

fim = time.time()
if show:
    print("Tempo decorrido para o cálculo dos 2 campos e gráficos: ", fim - inicio, "segundos\n\n")

plt.tight_layout()
figures.append(fig)

# Ganho (broadside)
def Gain():
    Kef = (2 * np.pi * flm_des) * np.sqrt(es * u0 * (1-1j*tgefTot(kLM, L, M)))
    K01 = 2*np.pi*flm[0][1]*np.sqrt(u0*es)
    K10 = 2*np.pi*flm[1][0]*np.sqrt(u0*es)
    K0 = kLM / np.sqrt(er)

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

    if M == 0:
        ki = Ckth * (R(np.cos(theta1c),1,0) * R(np.cos(thetap),1,0) / (np.abs(Kef**2-K10**2)*I10))**2
        if show:
            print('P10 =', Ckth * (2 * np.pi * flm_des * u0 / (Dphi * a_bar**2))**2 * (R(np.cos(theta1c),1,0) * R(np.cos(thetap),1,0) / (np.abs(Kef**2-K10**2)*I10))**2)
    elif M == 1:
        ki = Ckph * (2 * R(np.cos((theta1c+theta2c)/2),0,1) * R(np.cos(thetap),0,1) * np.cos(M01*(phip[0] - phi1c)) * np.sinc(M01*Dphip[0]/(2*np.pi)) / (np.abs(Kef**2-K01**2)*I01))**2
        if show:
            print('P01 =', (2 * np.pi * flm_des * u0 / (Dphi * a_bar**2))**2 * Ckph * (2 * R(np.cos((theta1c+theta2c)/2),0,1) * R(np.cos(thetap),0,1) * np.cos(M01*(phip[0] - phi1c)) * np.sinc(M01*Dphip[0]/(2*np.pi)) / (np.abs(Kef**2-K01**2)*I01))**2)

    ki *= (2 * np.pi * flm_des * u0 / (Dphi * a_bar**2))**2 / np.real(ZLP(flm_des, 1, 1)/2)

    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if M == 0:
                Sthv = ((1j ** Ml) * (- b * sp.lpmn(m, l, 0)[1] * (IdP_1 + IdP_2) / (dH2_dr) \
                    + 1j * Mm**2 * sp.lpmn(m, l, 0)[0] * (IpP_1 + IpP_2) / (K0 * H2) ) * \
                    (phi2-phi1) * np.sinc(Mm*(phi2-phi1)/(2*np.pi)) / S_lm)
                Sthv = np.sum(np.dot(delm, Sthv)) / np.pi
                D = 2*np.pi/eta0 * (np.abs(Sthv)**2) / (Ckth)
            elif M == 1:
                Sphh = ((1j ** Ml) * (Mm**2 * b * I_th * sp.lpmn(m, l, 0)[0] / dH2_dr \
                    + 1j * sp.lpmn(m, l, 0)[1] * -I_dth / (K0 * H2) ) * \
                    2 * deltaPhi * np.sinc(Mm*deltaPhi/(2 * np.pi)) * np.cos(Mm*(phi2-phi1+deltaPhi)/2) / S_lm)
                Sphh = np.sum(np.dot(delm, Sphh)) / np.pi
                D = 2*np.pi/eta0 * (np.abs(Sphh)**2) / (Ckph)

    G = ki * D
    
    if show:
        print('ki = ', ki)
        print('D = ', D, ', dB: ', 10*np.log10(D))
        print('G = ', G, ', dB: ', 10*np.log10(G))
        print('Ganho simulado (db): ', GainLin)
    
    parametros = f"Modelo:\nDiretividade (dB) = {10*np.log10(D)}\nEficiencia (%) = {100*ki[0]}\nGanho (dBi) = {10*np.log10(G)}\n\n"
    if M == 0:
        parametros += f"Simulacao:\nDiretividade (dB) = {10*np.log10(10**(GainLin/10)/0.864511)}\nEficiencia (%) = {86.4511}\nGanho (dBi) = {GainLin}" 
    elif M == 1:
        parametros += f"Simulacao:\nDiretividade (dB) = {10*np.log10(10**(GainLin/10)/0.855087)}\nEficiencia (%) = {85.5087}\nGanho (dBi) = {GainLin}"
    out_dir = 'Resultados'
    out_file = os.path.join(out_dir, 'LP'+str(1-M)+str(M)+'_Ganho.txt')
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, 'w') as f:
        f.write(parametros)

Gain()

for i, fig in enumerate(figures):
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.eps'), format = 'eps')
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.png'))
fim_total = time.time()
print("Tempo total para o fim do código: ", fim_total - inicio_total, "segundos\n")

if show:
    figSm.show()
    plt.show()