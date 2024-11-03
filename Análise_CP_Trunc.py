import numpy as np
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
import pandas as pd
import os
from sympy import symbols, Eq, solve

filein = input("Digite o nome do arquivo (ex: 'CP_Trunc_Param'): ")
inicio_total = time.time()
show = 0

parametros = {line.split(' = ')[0]: float(line.split(' = ')[1]) for line in open('Resultados/Param/' + filein + '.txt')}
Dthetaa, Dphia, thetap, phip, alphac, At, flm_des = [parametros[key] for key in ['Dthetaa', 'Dphia', 'thetap', 'phip', 'alphac', 'At', 'f']]

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

theta1 = np.pi/2 - Dthetaa/2 + dtc      # Ângulo de elevação 1 da cavidade (rad)
theta2 = np.pi/2 + Dthetaa/2 + dtc      # Ângulo de elevação 2 da cavidade (rad)
phi1 = np.pi/2 - Dphia/2 + dpc          # Ângulo de azimutal 1 da cavidade (rad)
phi2 = np.pi/2 + Dphia/2 + dpc          # Ângulo de azimutal 2 da cavidade (rad)

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

path = 'HFSS/CP_Trunc/'       # Resultados
output_folder = 'Resultados/Analise_CP_Trunc'
os.makedirs(output_folder, exist_ok=True)
figures = []

def Geotruc():
    P2 = np.array([b*np.cos(phi2)*np.sin(theta1), b*np.sin(phi2)*np.sin(theta1), b*np.cos(theta1)])
    
    alphay = np.arctan(1/(np.tan(theta1)*np.cos(phi1)))

    alphaa = np.arccos(-np.sin(theta1)**2*np.cos(2*phi1)-np.cos(theta1)**2)
    alphat = alphaa-alphac

    print('\n\nalphay, alphaa, alphat: ', alphay/dtr, alphaa/dtr, alphat/dtr)

    # Interseção com a lateral: 
    ul = -b/np.sqrt(1+np.tan((np.pi+alphat)/2)**2+((np.tan((np.pi+alphat)/2)/np.tan(phi2)-np.cos(alphay))/np.sin(alphay))**2)
    vl = ul*(np.tan((np.pi+alphat)/2)/np.tan(phi2)-np.cos(alphay))/np.sin(alphay)

    Dthetat = np.arccos((P2[0]*(ul*np.cos(alphay)+vl*np.sin(alphay)) + P2[1]*(ul*np.tan((np.pi+alphat)/2)) + P2[2]*(vl*np.cos(alphay)-ul*np.sin(alphay)))/b**2)

    print('Dthetat (analítico/aproximado): ', Dthetat/dtr, alphac*np.sin(alphay)/dtr)

    # Interseção com lado superior:
    ur = symbols('ur')

    equation = Eq(ur**2*(1+np.tan((np.pi+alphat)/2)**2)+(b*np.cos(theta1)/np.cos(alphay)+ur*np.tan(alphay))**2-b**2, 0)

    solutions = solve(equation, ur)
    ur = float(np.min(solutions))
    vr = b*np.cos(theta1)/np.cos(alphay)+ur*np.tan(alphay)

    Dphit = np.arccos((P2[0]*(ur*np.cos(alphay)+vr*np.sin(alphay)) + P2[1]*(ur*np.tan((np.pi+alphat)/2)))/(b*np.sin(theta1))**2)

    Aint = Dphit * b**2 * (1-np.cos(theta1))

    Dalpha = np.arccos(np.cos(theta1)*np.cos(theta1+Dthetat)+np.sin(theta1)*np.sin(theta1+Dthetat)*np.cos(Dphit))
    beta = np.pi-np.arcsin(np.sin(theta1+Dthetat)*np.sin(Dphit)/np.sin(Dalpha))
    gama = np.arcsin(np.sin(theta1)*np.sin(Dphit)/np.sin(Dalpha))
    Agirard = (Dphit+beta+gama-np.pi) * b**2
    At = Agirard-Aint

    print('Dphit (analítico/aproximado): ', Dphit/dtr, alphac*np.cos(alphay)/np.sin(theta1)/dtr)
    print('Área (mm²) (analítica/aproximada): ', At*1e6, b**2 * Dphit * np.cos((Dthetaa-Dthetat)/2) * np.sin(Dphit/2)*1e6, '\n')

if show:
    Geotruc()

# Dados do HFSS
data_hfss = pd.read_csv(path+'Z Parameter CP.csv')
freqs_hfss = data_hfss['Freq [GHz]'] # GHz
re_hfss = data_hfss['re(Z(1,1)) []']
im_hfss = data_hfss['im(Z(1,1)) []']
Zin_hfss = re_hfss + 1j*im_hfss

fig = plt.figure()
# Simulados
plt.plot(freqs_hfss, re_hfss, label='Re(Z(1,1))', color = 'r')
plt.plot(freqs_hfss, im_hfss, label='Im(Z(1,1))', color = '#8B0000')
plt.plot(freqs_hfss, [Z0] * len(freqs_hfss), label='Z0', color = 'y')
# Modelo analítico
plt.axvline(x=flm_des / 1e9, color='b', linestyle='--')
plt.title('Impedância na Simulação')
plt.xlabel('Frequência (GHz)')
plt.ylabel(r'Impedância ($\Omega$)')
plt.legend()
plt.grid(True)
figures.append(fig)

# Carta de Smith
figSm = go.Figure()
Zchart = (re_hfss+1j*im_hfss)/Z0
figSm.add_trace(go.Scattersmith(imag=np.imag(Zchart).tolist(), real=np.real(Zchart).tolist(), marker_color="red", name="Zin simulado"))
figSm.write_image(output_folder+"/Smith.png")

# Gamma_in = s11
fig = plt.figure()
plt.plot(freqs_hfss, 20*np.log10(np.abs((Zin_hfss-Z0)/(Zin_hfss+Z0))), label=r'$|s_{11}|$')
plt.axhline(y=-10, color='r', linestyle='--')
plt.axvline(x=flm_des / 1e9, color='b', linestyle='--')
plt.xlabel('Frequência (GHz)')
plt.ylabel(r'$|s_{11}|$' + ' (dB)')
plt.title('Coeficiente de reflexão (' + r'$|\Gamma_{in}|$' + ')')
plt.legend(loc='lower right')
plt.grid(True)
figures.append(fig)


# Razão Axial
def plotRA():
    data_hfss = pd.read_csv(path+'Axial Ratio Plot f.csv')
    freqs_hfss = data_hfss['Freq [GHz]'] # GHz
    ra_hfss = data_hfss['dB(AxialRatioValue) [] - Phi=\'90.0000000000002deg\' Theta=\'90.0000000000002deg\'']

    fig = plt.figure()
    plt.plot(freqs_hfss[7:25], ra_hfss[7:25])
    plt.axvline(x=flm_des / 1e9, color='r', linestyle='--')
    plt.axhline(y=3, color='g', linestyle='--')
    plt.xlabel('Frequência (GHz)')
    plt.ylabel('Razão Axial (dB)')
    plt.title('Razão Axial na Simulação')
    plt.grid(True)
    figures.append(fig)

    data_hfss = pd.read_csv(path+'Axial Ratio Plot Phi90.csv')
    angs_hfss = data_hfss['Theta [deg]'] # deg
    ra_hfss = data_hfss['dB(AxialRatioValue) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']

    fig = plt.figure()
    plt.plot(angs_hfss[91:], ra_hfss[91:])
    plt.axvline(x=90, color='r', linestyle='--')
    plt.axhline(y=3, color='g', linestyle='--')
    plt.xlabel('Ângulo '+r'$\theta$'+' (graus)')
    plt.ylabel('Razão Axial (dB)')
    plt.title('Razão Axial')
    plt.grid(True)
    figures.append(fig)

    data_hfss = pd.read_csv(path + 'Axial Ratio Plot Theta90.csv')
    angs_hfss = data_hfss['Phi [deg]'] # deg
    ra_hfss = data_hfss['dB(AxialRatioValue) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']

    fig = plt.figure()
    plt.plot(angs_hfss[91:], ra_hfss[91:])
    plt.axvline(x=90, color='r', linestyle='--')
    plt.axhline(y=3, color='g', linestyle='--')
    plt.xlabel('Ângulo '+r'$\varphi$'+' (graus)')
    plt.ylabel('Razão Axial (dB)')
    plt.title('Razão Axial')
    plt.grid(True)
    figures.append(fig)

plotRA()

# Campos distantes
eps = 1e-30

# Campos Ephi e Etheta
angulos = np.arange(0,360,1) * dtr + eps
angles = np.arange(0, 360, 30)
angles_labels = ['0°', '30°', '60°', '90°', '120°', '150°', '180°', '-150°', '-120°', '-90°', '-60°', '-30°']

fig, axs = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': 'polar'})

dataTh = pd.read_csv(path+'Gain Plot in theta.csv')
angTh = dataTh['Phi [deg]'] * dtr + eps
gainTh_P = dataTh['dB(GainPhi) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
Gsim1 = np.max(gainTh_P)
gainTh_P -= np.max(gainTh_P)
gainTh_T = dataTh['dB(GainTheta) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
Gsim2 = np.max(gainTh_T)
gainTh_T -= np.max(gainTh_T)
dataPh = pd.read_csv(path+'Gain Plot in phi.csv')
angPh = dataPh['Theta [deg]'] * dtr + eps
gainPh_T = dataPh['dB(GainTheta) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']
Gsim3 = np.max(gainPh_T)
gainPh_T -= np.max(gainPh_T)
gainPh_P = dataPh['dB(GainPhi) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']
Gsim4 = np.max(gainPh_P)
if show:
    print('Ganhos em cada componente (Ephi/Etheta):', Gsim1, Gsim2, '\n\n')
gainPh_P -= np.max(gainPh_P)

# Plotar o primeiro gráfico
axs[0, 0].plot(angTh, gainTh_T, color='red', label = 'Simulação')
axs[0, 0].set_title('Amplitude do Campo Elétrico TM10 (dB)')
axs[0, 0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0, 0].grid(True)
axs[0, 0].set_theta_zero_location('N')
axs[0, 0].set_theta_direction(-1)
axs[0, 0].set_rlim(-30,0)
axs[0, 0].set_thetagrids(angles)
axs[0, 0].set_rlabel_position(45)

# Plotar o segundo gráfico
axs[0, 1].plot(angTh, gainTh_P, color='red', label = 'Simulação')
axs[0, 1].set_title('Amplitude do Campo Elétrico TM01 (dB)')
axs[0, 1].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0, 1].grid(True)
axs[0, 1].set_theta_zero_location('N')
axs[0, 1].set_theta_direction(-1)
axs[0, 1].set_rlim(-30,0)
axs[0, 1].set_thetagrids(angles)
axs[0, 1].set_rlabel_position(45)

# Plotar o terceiro gráfico
axs[1, 0].plot(angPh, gainPh_T, color='red', label = 'Simulação')
axs[1, 0].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[1, 0].grid(True)
axs[1, 0].set_theta_zero_location('N')
axs[1, 0].set_theta_direction(-1)
axs[1, 0].set_rlim(-30,0)
axs[1, 0].set_thetagrids(angles, labels=angles_labels)
axs[1, 0].set_rlabel_position(45)

# Plotar o quarto gráfico
axs[1, 1].plot(angPh, gainPh_P, color='red', label = 'Simulação')
axs[1, 1].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[1, 1].grid(True)
axs[1, 1].set_theta_zero_location('N')
axs[1, 1].set_theta_direction(-1)
axs[1, 1].set_rlim(-30,0)
axs[1, 1].set_thetagrids(angles, labels=angles_labels)
axs[1, 1].set_rlabel_position(45)

handles, figlabels = axs[0, 1].get_legend_handles_labels()
fig.legend(handles, figlabels, loc='lower center', ncol=1)
plt.tight_layout()
figures.append(fig)

# Campos RHCP e LHCP
angulos = np.arange(0,360,1) * dtr + eps

fig, axs = plt.subplots(2, 2, figsize=(8, 8), subplot_kw={'projection': 'polar'})

dataTh = pd.read_csv(path+'Gain Plot HCP theta.csv')
angTh = dataTh['Phi [deg]'] * dtr + eps
gainTh_T = dataTh['dB(GainRHCP) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
G_RHCP = np.max(gainTh_T)
gainTh_T -= G_RHCP
gainTh_P = dataTh['dB(GainLHCP) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\'']
G_LHCP = np.max(gainTh_P)
gainTh_P -= G_RHCP
dataPh = pd.read_csv(path+'Gain Plot HCP phi.csv')
angPh = dataPh['Theta [deg]'] * dtr + eps
gainPh_T = dataPh['dB(GainRHCP) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']
gainPh_T -= G_RHCP
gainPh_P = dataPh['dB(GainLHCP) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\'']
gainPh_P -= G_RHCP
gTest = np.max(gainPh_P)

# Plotar o primeiro gráfico
axs[0, 0].plot(angTh, gainTh_T, color='red', label = 'Simulação')
axs[0, 0].set_title('Amplitude do Campo Elétrico RHCP')
axs[0, 0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0, 0].grid(True)
axs[0, 0].set_theta_zero_location('N')
axs[0, 0].set_theta_direction(-1)
axs[0, 0].set_rlim(-30,0)
axs[0, 0].set_thetagrids(angles)
axs[0, 0].set_rlabel_position(45)

# Plotar o segundo gráfico
axs[0, 1].plot(angTh, gainTh_P, color='red', label = 'Simulação')
axs[0, 1].set_title('Amplitude do Campo Elétrico LHCP')
axs[0, 1].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0, 1].grid(True)
axs[0, 1].set_theta_zero_location('N')
axs[0, 1].set_theta_direction(-1)
axs[0, 1].set_rlim(-30,0)
axs[0, 1].set_thetagrids(angles)
axs[0, 1].set_rlabel_position(45)

# # Plotar o terceiro gráfico
axs[1, 0].plot(angPh, gainPh_T, color='red', label = 'Simulação')
axs[1, 0].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[1, 0].grid(True)
axs[1, 0].set_theta_zero_location('N')
axs[1, 0].set_theta_direction(-1)
axs[1, 0].set_rlim(-30,0)
axs[1, 0].set_thetagrids(angles, labels=angles_labels)
axs[1, 0].set_rlabel_position(45)

# # Plotar o quarto gráfico
axs[1, 1].plot(angPh, gainPh_P, color='red', label = 'Simulação')
axs[1, 1].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[1, 1].grid(True)
axs[1, 1].set_theta_zero_location('N')
axs[1, 1].set_theta_direction(-1)
axs[1, 1].set_rlim(-30,0)
axs[1, 1].set_thetagrids(angles, labels=angles_labels)
axs[1, 1].set_rlabel_position(45)

handles, figlabels = axs[0, 1].get_legend_handles_labels()
fig.legend(handles, figlabels, loc='lower center', ncol=1)
plt.tight_layout()
figures.append(fig)

# Método da antena linear girante
normalizado = 0

datakRA = pd.read_csv(path+'kRA_CP.csv')
datakRA = datakRA[datakRA['Phi[deg]'] == 90]
magkRA = datakRA['mag(rETheta/rEphi)'].to_numpy()
angkRA = datakRA['ang_rad(rETheta/rEPhi)[rad]'].to_numpy()

dataPh = pd.read_csv(path+'Gain Plot in phi.csv')
angPh = dataPh['Theta [deg]'].to_numpy() * dtr + eps
gainPh_P = dataPh['dB(GainPhi) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\''].to_numpy()

corr = 5.5
if normalizado:
    corr = 0.7
vel  = 50
Eg_hfss_Ph = gainPh_P+10*np.log10(magkRA**2 *np.cos(vel*angPh)**2+np.cos(vel*angPh+angkRA)**2)+0.5
if normalizado:
    Eg_hfss_Ph -= np.max(Eg_hfss_Ph)-corr
    gainPh_P -= np.max(gainPh_P)

gainPh_T = dataPh['dB(GainTheta) [] - Freq=\'1.575GHz\' Phi=\'90.0000000000002deg\''].to_numpy()
if normalizado:
    gainPh_T -= np.max(gainPh_T)

datakRA = pd.read_csv(path+'kRA_CP.csv')
datakRA = datakRA[datakRA['Theta[deg]'] == 90]
magkRA = datakRA['mag(rETheta/rEphi)'].to_numpy()
angkRA = datakRA['ang_rad(rETheta/rEPhi)[rad]'].to_numpy()

dataTh = pd.read_csv(path+'Gain Plot in theta.csv')
angTh = dataTh['Phi [deg]'].to_numpy() * dtr + eps
gainTh_P = dataTh['dB(GainPhi) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\''].to_numpy()

vel  = 50
Eg_hfss_Th = gainTh_P+10*np.log10(magkRA**2 *np.cos(vel*angTh)**2+np.cos(vel*angTh+angkRA)**2)+0.5
if normalizado:
    Eg_hfss_Th -= np.max(Eg_hfss_Th)-corr
    gainTh_P -= np.max(gainTh_P)

gainTh_T = dataTh['dB(GainTheta) [] - Freq=\'1.575GHz\' Theta=\'90.0000000000002deg\''].to_numpy()
if normalizado:
    gainTh_T -= np.max(gainTh_T)

angulos = np.arange(0,360,1) * dtr + eps
fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'})

axs[0].plot(angPh, gainPh_P, label = r'$E_\varphi^H(\theta,\varphi)$')
axs[0].plot(angPh, gainPh_T, label = r'$E_\theta^V(\theta,\varphi)$')
axs[0].plot(angPh, Eg_hfss_Ph, color='red', label = 'Simulação')
axs[0].set_xlabel('Ângulo ' + r'$\theta$' + ' para ' + r'$\varphi = \frac{\varphi_{1c}+\varphi_{2c}}{2}$')
axs[0].grid(True)
axs[0].set_theta_zero_location('N')
axs[0].set_theta_direction(-1)
axs[0].set_rlim(-30,corr)
axs[0].set_thetagrids(angles, labels=angles_labels)
axs[0].set_rlabel_position(45)

axs[1].plot(angTh, gainTh_P, label = r'$E_\theta^V(\theta,\varphi)$')
axs[1].plot(angTh, gainTh_T, label = r'$E_\varphi^H(\theta,\varphi)$')
axs[1].plot(angTh, Eg_hfss_Th, color='red', label = 'Simulação')
axs[1].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[1].grid(True)
axs[1].set_theta_zero_location('N')
axs[1].set_theta_direction(-1)
axs[1].set_rlim(-30,corr)
axs[1].set_thetagrids(angles)
axs[1].set_rlabel_position(45)

fig.suptitle('Método da antena linear girante')
handles, figlabels = axs[0].get_legend_handles_labels()
fig.legend(handles, figlabels, loc='lower center', ncol=1)
figures.append(fig)
plt.tight_layout()

# Salvamento das figuras
for i, fig in enumerate(figures):
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.eps'), format = 'eps')
    fig.savefig(os.path.join(output_folder, f'figure_{i+1}.png'))
fim_total = time.time()
print("Tempo total para o fim do código: ", fim_total - inicio_total, "segundos\n")

if show:
    figSm.show()
    plt.show()