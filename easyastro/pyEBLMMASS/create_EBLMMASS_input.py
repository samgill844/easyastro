import uncertainties.unumpy as np
import os, sys
from astropy import constants
constants.pi = 3.14159265359

def create_EBLMMASS_input(Teff, FeH, radius_1, b, f_c, f_s, period, K, 
		step_size_scale = 0.5, MCMC_step_size_scaling_factor = 1, steps_for_MCMC_burn_in = 100000, steps_for_MCMC = 10000, thinningh_factor = 1, grid=1):
	
	# Calculate FWHM
	fwhm = (radius_1)*np.sqrt(1-b**2)/constants.pi
	print('FWHM: {}'.format(fwhm))

	# Calculate Mass function
	e  =f_c*f_c + f_s*f_s
	mass_func = ((1-e**2)**1.5)*period*86400.1*((K*10**3)**3)/(2*constants.pi*constants.G.value*1.989e30)
	print('Mass function: {}'.format(mass_func))

	# Open and write inpput file
	f = open('pyEBLMMASS_input.txt', 'wb')
	
	f.write(bytes('{}\n'.format(grid), 'utf-8'))
	f.write(bytes('{:.1f} {} {}\n'.format(Teff.nominal_value,124,-124), 'utf-8')) # Teff
	f.write(bytes('{} {} {}\n'.format(0, 5, -5), 'utf-8')) # logl/Lsun 
	f.write(bytes('{:.1f} {} {}\n'.format(FeH.nominal_value,0.14,-0.14), 'utf-8')) # [Fe/H]s
	f.write(bytes('{:.5f} {:.5f} {:.5f}\n'.format(radius_1.nominal_value,radius_1.std_dev,-radius_1.std_dev), 'utf-8')) # R_*/a
	f.write(bytes('{:.5f} {:.5f} {:.5f}\n'.format(mass_func.nominal_value,mass_func.std_dev,-mass_func.std_dev), 'utf-8')) # mass func
	f.write(bytes('{:.5f} {:.5f} {:.5f}\n'.format(fwhm.nominal_value,fwhm.std_dev,-fwhm.std_dev), 'utf-8')) # fwhm

	f.write(bytes('{} {} {} {}\n'.format(0.0,   0.0,  17.5,   0.0), 'utf-8')) # age prior
	f.write(bytes('{} {} {} {}\n'.format(0.00, -0.75,  0.55,  0.00), 'utf-8')) # prior on surface Fe/H
	f.write(bytes('{}\n'.format(0.00), 'utf-8')) # mass func prior
	f.write(bytes('{}\n'.format(period.nominal_value), 'utf-8')) # Period

	f.write(bytes('{}\n'.format(step_size_scale), 'utf-8')) # step size scale
	f.write(bytes('{}\n'.format(MCMC_step_size_scaling_factor), 'utf-8')) # MCMC step size scaling factor
	f.write(bytes('{}\n'.format(steps_for_MCMC_burn_in), 'utf-8')) # steps for MCMC burn-in
	f.write(bytes('{}\n'.format(steps_for_MCMC), 'utf-8')) # steps for MCMC 
	f.write(bytes('{}\n'.format(thinningh_factor), 'utf-8')) # Thinning factor for output chain
	f.close()
	

		
	# now make the call
	#os.system("eblmmass < pyEBLMMASS_input.txt")
