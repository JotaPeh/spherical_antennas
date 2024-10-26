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

L, M = 0, 1             # Modo TM_LM em teste, TM10 ou TM01 apenas

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
theta1c = 67.3524873620928 * dtr + dtc       # Ângulo de elevação 1 da cavidade (rad)
theta2c = 112.6475126379072 * dtr + dtc      # Ângulo de elevação 2 da cavidade (rad)
phi1c = 72.57883643237908 * dtr + dpc        # Ângulo de azimutal 1 da cavidade (rad)
phi2c = 107.42116356762092 * dtr + dpc       # Ângulo de azimutal 2 da cavidade (rad)

Dtheta = theta2c - theta1c
Dphi = phi2c - phi1c
deltatheta1 = h/a       # Largura angular 1 do campo de franjas e da abertura polar (rad)
deltatheta2 = h/a       # Largura angular 2 do campo de franjas e da abertura polar (rad)
theta1, theta2 = theta1c + deltatheta1, theta2c - deltatheta2 # Ângulos polares físicos do patch (rad)
deltaPhi = h/a          # Largura angular do campo de franjas e da abertura azimutal (rad)
phi1, phi2 = phi1c + deltaPhi, phi2c - deltaPhi             # Ângulos azimutais físicos do patch (rad)

# def test(x):
#     return x**np.sin(x-1)

# u = np.linspace(0.0,4.0,41)
# print(u)
# print(test(u))
# print(np.vectorize(test)(u))

# Em dB
# Pmin = np.array([-45.9, -46.4, -15.6, -19.3, -40.8])
# Psub = np.array([-39.9, -38.4,  -7.6,  -9.4,   -37])

# delta = Pmin - Psub

# VSWR = (1+10**(delta/20))/(1-10**(delta/20))

# print('VSWR = ', VSWR)

angulos = np.arange(0,360,1) * dtr + eps
angles = np.arange(0, 360, 30)
angles_labels = ['0°', '30°', '60°', '90°', '120°', '150°', '180°', '-150°', '-120°', '-90°', '-60°', '-30°']

fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'})

dataTh = pd.read_csv('Ant01_Theta90.csv')
angTh = dataTh['Phi [deg]'] * dtr + eps
gainTh = dataTh['dB(GainPhi) [] - Freq=\'1.603GHz\' Theta=\'90deg\'']
gainTh -= np.max(gainTh)
dataPh = pd.read_csv('Ant01_Phi90.csv')
angPh = dataPh['Theta [deg]'] * dtr + eps
gainPh = dataPh['dB(GainPhi) [] - Freq=\'1.603GHz\' Phi=\'90deg\'']
gainPh -= np.max(gainPh)
print(angTh, gainTh, angulos)

axs[0].plot(angTh, gainTh, color='red', label = 'Simulação') #dado fake
axs[0].set_title('Plano E') # A confirmar
axs[0].set_xlabel('Ângulo ' + r'$\varphi$' + ' para ' + r'$\theta = \frac{\theta_{1c}+\theta_{2c}}{2}$')
axs[0].grid(True)
axs[0].set_theta_zero_location('N')
axs[0].set_theta_direction(-1)
axs[0].set_rlim(-30,0)
axs[0].set_thetagrids(angles, labels=angles_labels)
axs[0].set_rlabel_position(45)

axs[1].plot(angPh, gainPh, color='red', label = 'Simulação') #dado fake
axs[1].set_title('Plano H') # A confirmar
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

plt.tight_layout()
plt.show()