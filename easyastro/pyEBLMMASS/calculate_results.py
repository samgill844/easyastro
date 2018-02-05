import corner
import numpy as np
from astropy.table import Table, Column
from uncertainties import *
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def calculate_results(t,period, radius_1, k, return_corner=True, return_HR=True, use='median'):

	eblmmass = Table.read("chain.dat",format="ascii")

	if use=='best':
		best_index = np.argmax(np.array(eblmmass['loglike']))
		age = ufloat(eblmmass['Age'][best_index], np.std(eblmmass['Age']))
		M_1 = ufloat(eblmmass['M_*'][best_index], np.std(eblmmass['M_*']))
		M_2 = ufloat(eblmmass['M_comp'][best_index], np.std(eblmmass['M_comp']))

	if use=='median':
		age = ufloat(np.median(eblmmass['Age']), np.std(eblmmass['Age']))
		M_1 = ufloat(np.median(eblmmass['M_*']), np.std(eblmmass['M_*']))
		M_2 = ufloat(np.median(eblmmass['M_comp']), np.std(eblmmass['M_comp']))

	a = 4.20944009361 *period**(2./3.) * (M_1+M_2)**(1./3.)
	R_1 = radius_1*a
	R_2 = radius_1*k*a

	eblmmass.add_column(Column(t['radius_1']*a.nominal_value,   name = 'R_*'))
	eblmmass.add_column(Column(t['radius_1']*t['k']*a.nominal_value,   name = 'R_comp'))

	print('\n\nPrimary:')
	print("\tM_1 = {:.3f} M_Sun".format(M_1))
	print("\tR_1 = {:.3f}  R_Sun".format(R_1))
	print('Secondary:')
	print("\tM_2 = {:.3f} M_Sun".format(M_2))
	print("\tR_2 = {:.3f}  R_Sun".format(R_2))

	print('Parameters:')
	print("\tAge = {:.3f} Gyr".format(age))
	roche = 1.26*radius_1*(M_1/M_2)**(1/3)
	print('\tRoch limit = {:.3f} R_Sun'.format(roche))
	print('\ta = {:.3f}'.format(a))

	if return_corner:
		print('\nCreating corner plot')
		tmp = np.array([eblmmass['M_*'], eblmmass['R_*'], eblmmass['M_comp'], eblmmass['R_comp'], eblmmass['Age']]).T
		corner_fig = corner.corner(tmp, labels = [r'$\rm M_1$', r'$\rm R_1$', r'$\rm M_2$', r'$\rm R_2$', r'$ \rm Age$'],
			     truths = [M_1.nominal_value, R_1.nominal_value, M_2.nominal_value, R_2.nominal_value, age.nominal_value],
			     show_titles=True, title_kwargs={"fontsize": 12})
	
	if return_HR:
		print('\nCreating HR plot')
		zams = Table.read('zams.dat',format='ascii')
		isoch = Table.read('isochrone.dat',format='ascii')
		isochl = Table.read('isochrone_alo.dat',format='ascii')
		isochh = Table.read('isochrone_ahi.dat',format='ascii')

		x = eblmmass['Teff'][::500]  #teff
		y = eblmmass['M_*']/np.power(t['radius_1']*a.nominal_value,3) # rho
		y=y[::500]
		# Calculate the point density
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)

		# Sort the points by density, so that the densest points are plotted last
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]

		hr_fig, ax = plt.subplots()
		ax.scatter(x, y, c=z, s=10, edgecolor='',label='J2349-32')

		plt.plot(np.array(zams['col5']),np.array(zams['col1'])/np.power(np.array(zams['col2']),3),'g--')
		plt.plot(np.array(isoch['col5']),np.array(isoch['col1'])/np.power(np.array(isoch['col2']),3),'k')
		plt.plot(np.array(isochl['col5']),np.array(isochl['col1'])/np.power(np.array(isochl['col2']),3),'k--')
		plt.plot(np.array(isochh['col5']),np.array(isochh['col1'])/np.power(np.array(isochh['col2']),3),'k--')


		plt.gca().invert_yaxis();plt.gca().invert_xaxis()
		ax = plt.gca()
		ax.set_yscale('log')
		plt.xlabel(r'$\rm T_{\rm eff} \, \rm (K)$')
		plt.ylabel(r'$ \rho \, / \rho_{sol}$')
		plt.grid(alpha=0.1)
		
	if return_corner and not return_HR:
		return corner_fig
	if not return_corner and return_HR:
		return return_HR
	if return_corner and return_HR:
		return corner_fig,return_HR

