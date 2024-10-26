from scipy import special as sp
from scipy.optimize import root_scalar
from scipy import integrate as spi
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time
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

# Coordenadas dos alimentadores coaxiais (rad)
thetap = [90.0 * dtr, 95.0 * dtr]  
phip = [82.4 * dtr, 90.0 * dtr]
probes = len(thetap)    # Número de alimentadores
df = 1.3e-3             # Diâmetro do pino central dos alimentadores coaxiais (m)
er = 2.55               # Permissividade relativa do substrato dielétrico
es = e0 * er            # Permissividade do substrato dielétrico (F/m)
Dphi = [np.exp(1.5)*df/(2*a*np.sin(t)) for t in thetap]     # Comprimento angular do modelo da ponta de prova (rad)

# Outras variáveis
tgdel = 0.022           # Tangente de perdas
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

def Equation(l, theta1, theta2, m):
    z1 = np.cos(theta1)
    z2 = np.cos(theta2)
    return np.where(np.abs(z1) < 1 and np.abs(z2) < 1, DP(theta1,l,m)*DQ(theta2,l,m)-DQ(theta1,l,m)*DP(theta2,l,m) , 1)

Lambda = np.linspace(-0.1, 32, 64) # Domínio de busca, o professor usou a janela de 0.1
rootsLambda = []

def root_find(n):
    roots = []
    Wrapper = partial(Equation, theta1 = theta1c, theta2 = theta2c, m = n)
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
        print("n =", n, ":", [round(r, 5) for r in roots[k:k+5]], ', mu =', n * np.pi / (phi2c - phi1c), "\n\n")
    rootsLambda.append(roots[k:k+5])

if show:
    print('Raízes lambda:')
for n in range(0,6):
    root_find(n)

fim = time.time()
if show:
    print("Tempo decorrido:", fim - inicio_total, "segundos\n\n")

if show:
    # Equação para cada m
    m = 1
    Eq = Equation(Lambda, theta1c, theta2c, m)
    plt.figure()
    plt.plot(Lambda, Eq, label='m=1')
    plt.title('Equation')
    plt.grid(True)
    plt.show()
    
    # Raízes indesejadas
    print('Estudo das raízes indesejadas:')
    print(m * np.pi / (phi2c - phi1c))
    print(m * np.pi / (phi2c - phi1c) - 1.11363636363806)

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

def R(v,l,m):                                  # Função auxiliar para os campos elétricos
    L = rootsLambda[l][m]
    return (DP(theta1c,L,m)*legQ(v,L,m) - DQ(theta1c,L,m)*legP(v,L,m)) * np.sin(theta1c)

def Er_lm_norm(theta,phi,l,m):                 # Campo elétrico dos modos 'normalizados' da solução homogênea
    u = m * np.pi / (phi2c - phi1c)
    return R(np.cos(theta),l,m) * np.cos(u * (phi - phi1c))

phi = np.linspace(phi1c, phi2c, 200)           # Domínio de phi (rad)
theta = np.linspace(theta1c, theta2c, 200)     # Domínio de theta (rad)
L, M = 0, 1                                    # Modo em teste

P, T = np.meshgrid(phi, theta)
Er_LM = Er_lm_norm(T, P, L, M)
Amp = np.abs(Er_LM/np.max(np.abs(Er_LM)))      # Normalizado
Phase = np.angle(Er_LM)

if show:
    print('\nMáximo do campo', np.max(Er_LM))
    print('Modo testado: TM', L, M)
    print('Frequência do modo testado: ', flm[L][M] / 1e9)

    # Mapa de Amplitude
    plt.figure(figsize=(8, 6))
    plt.contourf(P / dtr, T / dtr, Amp, cmap='jet', levels = 300)
    plt.colorbar()
    plt.xlabel(r'$\varphi$' + ' (graus)', fontsize=14)
    plt.ylabel(r'$\theta$' + ' (graus)', fontsize=14)
    plt.title('Mapa de Amplitude (Normalizado)')
    plt.show()

    # Mapa de Fase
    plt.figure(figsize=(8, 6))
    plt.contourf(P / dtr, T / dtr, Phase / dtr, cmap='gnuplot', levels = 200)
    plt.colorbar(label = 'Fase (grau)')
    plt.xlabel(r'$\varphi$' + ' (graus)', fontsize=14)
    plt.ylabel(r'$\theta$' + ' (graus)', fontsize=14)
    plt.title('Mapa de Fase')
    plt.show()

# Impedância de entrada - Circuito RLC:
def R2(v, l, m):                               # Quadrado da função auxiliar para os campos elétricos
    return R(v, l, m)**2

def tgef(L,M):
    # Qdie = 2 * np.pi * f * es / sigma_die # + Perdas de irradiação = 1/tgdel
    Rs = np.sqrt(2 * np.pi * flm[L][M] * u0 / (2 * sigma))
    Qc = 2 * np.pi * flm[L][M] * u0 * h / (2 * Rs)
    Qc = Qc * (3*a**2 + 3*a*h + h**2) / (3*a**2 + 3*a*h + h**2 * 3/2)
    return tgdel + 1/Qc

def RLC(f, L, M, p1, p2):
    U = M * np.pi / (phi2c - phi1c)
    Ilm, error = spi.quad(partial(R2, l = L, m = M),np.cos(theta2c),np.cos(theta1c))

    # Auxiliar alpha
    alpha = h * (R(np.cos(thetap[p1-1]), L, M) * np.cos(U * (phip[p1-1]-phi1c)) * np.sinc(U * Dphi[p1-1] / 2)) * \
        (R(np.cos(thetap[p2-1]), L, M) * np.cos(U * (phip[p2-1]-phi1c)) * np.sinc(U * Dphi[p2-1] / 2)) / ((phi2c - phi1c) * es * a_bar**2 * Ilm)
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
    if isinstance(f, list) and len(f) > 1:     # Não usar f como vetor
        f = flm[L][M]
    Zmatrix = []
    for q in range(probes):  
        line = []
        for p in range(probes):
            line.append(Z(f,L,M,q+1,p+1))
        Zmatrix.append(line)
    return Zmatrix

def Slm(f, L, M):                              # Matriz de espalhamento
    return (np.linalg.inv(np.array(Zlm(f, L,M))/Z0 + np.eye(probes)) * (np.array(Zlm(f, L,M))/Z0 - np.eye(probes))).tolist()

freqs01 = np.linspace(1.5, 1.7, 641) * 1e9
freqs10 = np.linspace(1.1, 1.2, 641) * 1e9

if show:
    print('\nMatriz Z: ', Zlm(flm[0][1],0,1))
    print('\nMatriz S: ', Slm(flm[0][1],0,1))

    # Modo TM01 com a ponta de prova 1
    print('\nTM01: ', flm[0][1] / 1e9)
    plt.figure()
    plt.plot(freqs01 / 1e9, np.real(Z(freqs01, 0, 1, 1, 1)), label='re(Z)')
    plt.plot(freqs01 / 1e9, np.imag(Z(freqs01, 0, 1, 1, 1)), label='im(Z)')
    plt.plot(freqs01 / 1e9, [Z0] * len(freqs01), label='Z0')
    plt.title('Impedance')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Modo TM10 com a ponta de prova 2
    print('TM10: ', flm[1][0] / 1e9)
    plt.figure()
    plt.plot(freqs10 / 1e9, np.real(Z(freqs10, 1, 0, 2, 2)), label='re(Z)')
    plt.plot(freqs10 / 1e9, np.imag(Z(freqs10, 1, 0, 2, 2)), label='im(Z)')
    plt.plot(freqs10 / 1e9, [Z0] * len(freqs10), label='Z0')
    plt.title('Impedance')
    plt.legend()
    plt.grid(True)
    plt.show()

# Impedâncias de entrada, considerando Iin1 = Iin2
freqs = np.linspace(1.1, 1.7, 641) * 1e9
Iin = [1, 1]
Zin1 = Z(freqs,0,1,1,1) + Z(freqs,0,1,1,2) * Iin[1] / Iin[0]
Zin2 = Z(freqs,0,1,2,1) * Iin[0] / Iin[1] + Z(freqs,0,1,2,2)

if show:
    plt.figure()
    plt.plot(freqs / 1e9, np.real(Zin1), label='re(Zin1)')
    plt.plot(freqs / 1e9, np.imag(Zin1), label='im(Zin1)')
    plt.plot(freqs / 1e9, [Z0] * len(freqs), label='Z0')
    plt.title('Impedance')
    plt.legend()
    plt.grid(True)
    plt.show()

# Carta de Smith
if show:
    fig = go.Figure()
    Zchart = Z(freqs, 0, 1, 1, 1)/Z0 # Multiplicar fora do argumento da função abaixo
    fig.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="red", name="Z11"))
    Zchart = Z(freqs, 1, 0, 2, 2)/Z0
    fig.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="green", name="Z22"))
    # Zchart = Zin1/Z0
    # fig.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), name="Zin2"))
    fig.show()

# Dados do HFSS
data_hfss = pd.read_csv('Z Parameter Plot 1_compare.csv')
freqs_hfss = data_hfss['Freq [GHz]'] # GHz
re_hfss = data_hfss['re(Z(1,1)) []']
im_hfss = data_hfss['im(Z(1,1)) []']

if show:
    plt.figure()
    # Simulados
    plt.plot(freqs_hfss, re_hfss, label='re(Z(1,1) simulado')
    plt.plot(freqs_hfss, im_hfss, label='im(Z(1,1)) simulado')
    plt.plot(freqs_hfss, [Z0] * len(freqs_hfss), label='Z0')
    # Teóricos
    plt.plot(freqs_hfss, np.real(Z(freqs_hfss * 1e9, 0, 1, 1, 1)), label='re(Z(1,1) teórico')
    plt.plot(freqs_hfss, np.imag(Z(freqs_hfss * 1e9, 0, 1, 1, 1)), label='im(Z(1,1)) teórico')
    plt.title('Impedance')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Impedance')
    plt.legend()
    plt.grid(True)
    plt.show()

# Campos distantes do modo TM01
def hankel_spher2(n, x, der = 0):
    return sp.spherical_jn(n, x, der) - 1j * sp.spherical_yn(n, x, der)

def schelkunoff2(n, x, der = 1):
    return sp.riccati_jn(n, x)[der] - 1j * sp.riccati_yn(n, x)[der]

def Itheta(theta, L, M):
    return sp.lpmn(M, L, np.cos(theta))[0][M, L] * np.sin(theta)

def IDtheta(theta, L, M):
    return sp.lpmn(M, L, np.cos(theta))[1][M, L] * np.sin(theta)

def Eth_v_prot(f, ML, MM, theta, phi):
    Eth0_A, Eth0_C = 1, 1
    k = 2 * np.pi * f / c

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
    k = 2 * np.pi * f / c

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
    k = 2 * np.pi * f / c
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
    u = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eph_v_prot)(f, ML, MM, theta, phi))]))
    v = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eth_v_prot)(f, ML, MM, theta, phi))]))
    tot = v**2 + u**2
    # tot = v
    return 20*np.log10(tot/np.max(tot[~np.isnan(tot)]))

def E_h(f, ML, MM, theta, phi):
    u = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eph_h_prot)(f, ML, MM, theta, phi))]))
    v = np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eth_h_prot)(f, ML, MM, theta, phi))]))
    tot = u**2 + v**2
    # tot = u
    return 20*np.log10(tot/np.max(tot[~np.isnan(tot)]))

if 1:
    inicio = time.time()
    angulos = np.arange(0,359,2) * dtr
    print(angulos)
    ML, MM, angs = np.meshgrid(np.arange(1,15), np.arange(1,15), angulos)
    print(np.abs(np.array([np.sum(p) for p in np.transpose(np.vectorize(Eph_v_prot)(flm[1][0], ML, MM, angs, 90 * dtr))])))
    fim = time.time()
    print("Tempo decorrido para o cálculo paralelo: ", fim - inicio, "segundos\n\n")

if show: #1:
    inicio = time.time()
    angulos = np.arange(0,359,2) * dtr
    ML, MM, angs = np.meshgrid(np.arange(1,15), np.arange(1,15), angulos)
    ML180, MM180, angs180 = np.meshgrid(np.arange(1,15), np.arange(1,15), np.arange(0,179,2) * dtr)
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # Plotar o primeiro gráfico
    axs[0, 0].plot((angulos)[1:-1], E_v(flm[1][0], ML, MM, 90 * np.pi / 180, angs)[1:-1])
    axs[0, 0].set_title('Amplitude do Campo Elétrico TM10 (dB)')
    axs[0, 0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = 90$' + '°')
    axs[0, 0].grid(True)
    axs[0, 0].set_theta_zero_location('N')
    axs[0, 0].set_theta_direction(-1)

    # Plotar o segundo gráfico
    axs[0, 1].plot((angulos), E_h(flm[0][1], ML, MM, 90 * np.pi / 180, angs))
    axs[0, 1].set_title('Amplitude do Campo Elétrico TM01 (dB)')
    axs[0, 1].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = 90$' + '°')
    axs[0, 1].grid(True)
    axs[0, 1].set_theta_zero_location('N')
    axs[0, 1].set_theta_direction(-1)

    # Plotar o terceiro gráfico
    E1 = E_v(flm[1][0], ML180, MM180, angs180, (phi1c + phi2c) / 2)
    E2 = E_v(flm[1][0], ML180, MM180, angs180, -(phi1c + phi2c) / 2)
    axs[1, 0].plot((angulos), np.concatenate((E1, E2)), color='red')
    axs[1, 0].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
    axs[1, 0].grid(True)
    axs[1, 0].set_theta_zero_location('N')
    axs[1, 0].set_theta_direction(-1)

    # Plotar o quarto gráfico
    axs[1, 1].plot((angulos), E_h(flm[0][1], ML, MM, angs, (phi1c + phi2c) / 2), color='red')
    axs[1, 1].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
    axs[1, 1].grid(True)
    axs[1, 1].set_theta_zero_location('N')
    axs[1, 1].set_theta_direction(-1)
    
    fim = time.time()
    print("Tempo decorrido para o cálculo dos 4 campos e gráficos: ", fim - inicio, "segundos\n\n")

    plt.tight_layout()
    plt.show()
