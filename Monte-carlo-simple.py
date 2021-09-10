#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats

#Nombre de tirages simulés
N=100000                                                                         


#Variables d'entrée
#Il faut préciser la forme des variables d'entrée
Cb = np.random.uniform (9.9,10.1,N)
Vb = np.random.uniform (.099, .101,N)
Va = np.random.triangular(.099 ,.100 ,.101,N)
#Vb = np.random.normal(0.1, 0.01,N)

#print(Va)
Ca=Cb*Vb/Va
#écart-type réduit
uCa = np.std(Ca,ddof=1)                                                           
CaMoy= np.average(Ca)


Ccla=10
uCla=np.sqrt((0.1/(10*np.sqrt(3)))**2+(0.001/(np.sqrt(3)*0.1))+(0.001/(np.sqrt(6)*0.1)))

#abscisse
x = np.linspace(np.min(Ca),np.max(Ca),100)
Gauss = stats.norm.pdf(x, CaMoy, uCa)
GaussCla = stats.norm.pdf(x, Ccla, uCla)

plt.hist(Ca, bins='auto',density = True, label = "Monte-Carlo")
plt.plot(x, Gauss, 'k-', label = "Gaussienne associée")
plt.plot(x, GaussCla, 'k-',color='C1', label = "Gaussienne associée")
plt.title('Simulation de {} titrages \n Ca = {:.3e} mol/L \n u(Ca) = {:3e} {:3e}'.format(N,CaMoy,uCa,uCla))
plt.xlabel('Ca en mol/L$')
plt.legend()
plt.show()
