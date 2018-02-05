
import numpy as np
import ellc
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import corner, time as time_, sys, os
from astropy.table import Table, Column
from celerite.modeling import Model
from celerite import terms
import celerite
from easyastro.ellc_wrappers import get_ellc_params, get_ellc_bounds, get_ellc_priors, get_ellc_all, check_ellc_input, ellc_wrap,phaser, get_ellc_GP_params, ellc_GP_wrap,neg_log_like, log_probability
import types



def get_param(self):
	names = list(self.get_parameter_names())
	vals = self.get_parameter_vector()
	tmp = dict()
	for i in range(len(names)):
		tmp[names[i][5:]] = vals[i]
	return tmp


def calculate_log_like_prior(self, ellc_params,ellc_bounds, ellc_priors, specific_val = None ):
	# Now get prior
	param_keys = np.array([i for i in ellc_params.keys()])
	bound_keys = np.array([i for i in ellc_bounds.keys()])
	prior_keys = np.array([i for i in ellc_priors.keys()])
	params_with_bounds = list(set( [i for i in ellc_params.keys()]).intersection( [i for i in ellc_bounds.keys()]))

	#print(params_with_bounds)

	if specific_val==None:          
		prior = int(0)
		for i in params_with_bounds:
			low, val, high = ellc_bounds[i][0], ellc_params[i],  ellc_bounds[i][1]
			if (val<low) or (val>high):
				print(i, low, val, high)
				return -np.inf
			else:
				if i in prior_keys:
			        	prior += (ellc_params[i] - ellc_priors[i][0])**2 / (ellc_priors[i][1])**2


		return -0.5*prior

	else:
		low, val, high = ellc_bounds[specific_val][0], ellc_params[specific_val],  ellc_bounds[specific_val][1]
		if (val<low) or (val>high):
			return np.inf
		else:
			if specific_val in prior_keys:
				return (ellc_params[specific_val] - ellc_priors[specific_val][0])**2 / (ellc_priors[specific_val][1])**2
			else:
				return int(0)  

 
	
	
		


def fit_lc(time, mag,mag_err, ellc_params, ellc_bounds, ellc_priors, plot_model = True, minimize_lc=True, emcee_lc=False, emcee_draws = 100,emcee_chain_file = 'emcee_chain.fits', GP=True, free_params = ['t_zero', 'k', 'radius_1','b'],not_included_params=[], return_handler=False, return_model=False, gp_kernel = terms.Matern32Term(log_sigma=-2, log_rho = 2), gp_kernel_bounds=dict(), gp_kernel_freeze=[]):
	if GP:
		##########################################################################
		# Ok so this is a bit tricky. We will do the following to please celerite
		#
		# 1. Create the model and 
		# 1. Define a subset of ellc_params called ellc_GP_params which are just
		# floats and integeres (exclude None and string entries).
		##########################################################################

		ellc_GP_params,not_included_params,ellc_gp_bounds = get_ellc_GP_params(ellc_params, ellc_bounds, free_params)
		lc_model = ellc_GP_wrap(**ellc_GP_params, bounds = ellc_gp_bounds)# not_included_params, ellc_gp_bounds)


		lc_model.ellc_GP_params, lc_model.not_included_params, lc_model.ellc_gp_bounds, lc_model.ellc_gp_priors  = ellc_GP_params,not_included_params,ellc_gp_bounds,ellc_priors

		gp = celerite.GP(gp_kernel, mean=lc_model, fit_mean=True)
		gp.compute(time, yerr = mag_err)
		lglike = gp.log_likelihood(mag)

		########################################
		# Freeze parameter we do ot wish to fit
		########################################
		all_parameters = np.array(tuple(ellc_GP_params.keys()))
		params_to_freeze = np.setdiff1d(all_parameters, free_params)

		for i in params_to_freeze:
			gp.freeze_parameter('mean:{}'.format(i))
		for i in gp_kernel_freeze:
			gp.freeze_parameter('kernel:{}'.format(i))



		if return_handler:
			gp.get_param =  types.MethodType(get_param, gp)
			gp.calculate_log_like_prior = types.MethodType(calculate_log_like_prior, gp)
			return gp


		if plot_model:
			plt.scatter(time, mag, c='k', s= 10, alpha=0.5)
			plt.plot(time, lc_model.get_lc(time), 'r', alpha=0.8, label='Mean model')

			# Plot the variance with upsampled time series
			mu, var = gp.predict(mag, np.linspace(min(time), max(time), \
			     len(time)*10), return_var=True)
			std = np.sqrt(abs(var))
			color = "#ff7f0e"
			plt.fill_between( np.linspace(min(time), max(time), len(time)*10), \
			     mu+std, \
			     mu-std, color=color, alpha=0.8, edgecolor="none", label = 'GP model')

		
			
			plt.legend()
			plt.gca().invert_yaxis()
			plt.ylabel('Mag')
			plt.xlabel('Time')

			print('~'*80)
			print('Initial fit')
			print('~'*80)
			print('Loglike: {:.2f}'.format(lglike))
			print('Chi**2: {:.2f}'.format(-2*lglike))
			print('Reduced Chi**2: {:.2f}'.format(-2*lglike / (len(time) - len(free_params) ) ))
			print('BIC: {:.2f}'.format(len(free_params)*np.log(len(time)) - 2*np.log(lglike)))
			print('~'*80)
			plt.show()

			return np.linspace(min(time), max(time), len(time)*10), mu, var

		if minimize_lc:
			plt.scatter(time, mag, c='k', s= 10, alpha=0.5)
			plt.plot(time, lc_model.get_lc(time), 'g', alpha=0.8, label='Old')
			#plt.gca().invert_yaxis()
			plt.ylabel('Mag')
			plt.xlabel('Time')

			print('Minimisation of lightcurve with GP')
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			print("{:>20}: {:.3f}".format('Initial log-like',gp.log_likelihood(mag)))

			print('\n\n')
			print('~'*80)
			print("Minization using Scipy's L-BFGS-R algorithm with GP")
			print('~'*80)
			print('\nRunning L-BFGS-s algorithm . . .', end='') 


			##################
			# Now optimise
			##################
			initial_params = gp.get_parameter_vector()
			bounds_ = gp.get_parameter_bounds()

			soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds_, args=(mag, gp))
			print(soln)
			###################
			# Print the output
			###################
			print("{:>20}: {:.3f} ({} iterations)".format('Final log-like', -soln.fun, soln.nit))
			print('Chi**2: {:.2f}'.format(-2*-soln.fun))
			print('Reduced Chi**2: {:.2f}'.format(-2*-soln.fun / (len(time) - len(free_params) ) ))
			print('BIC: {:.2f}'.format(len(free_params)*np.log(len(time)) - 2*np.log(-soln.fun)))
			print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
			#for i in range(len(gp.get_parameter_names())):
			#	print('{:.3f}             {}'.format(soln.x[i], gp.get_parameter_names()[i]))

			############################################################################
			# Now get the best model as a fictionary so we can put it back in and plot
			############################################################################
			best_params = ellc_GP_params.copy()
			best_params_passback = ellc_params.copy()
			for i in range(len(gp.parameter_names)):
				if gp.parameter_names[i][:5]=='mean:':
					best_params[gp.parameter_names[i][5:]] = gp.parameter_vector[i]
					best_params_passback[gp.parameter_names[i][5:]] = gp.parameter_vector[i]
					#print('{:>12}:      {:.3f}'.format(gp.parameter_names[i][5:], gp.parameter_vector[i]))


			

			#########################
			# Now get the best model
			#########################
			plt.cla()

			# plot data
			plt.scatter(time,mag,s=2, c='k', alpha = 0.3)

			# Plot the best model (just to be sure)
			mean_model_better = ellc_GP_wrap(**best_params, bounds = ellc_gp_bounds)
			mean_model_better.ellc_GP_params, mean_model_better.not_included_params, mean_model_better.ellc_gp_bounds, mean_model_better.ellc_gp_priors  = ellc_GP_params,not_included_params,ellc_gp_bounds,ellc_priors
			plt.plot(time,mean_model_better.get_value(time), 'b--', linewidth=2, \
			     label='Model selected')

			# Plot the variance with upsampled time series
			mu, var = gp.predict(mag, np.linspace(min(time), max(time), \
			     len(time)*10), return_var=True)
			std = np.sqrt(abs(var))
			color = "#ff7f0e"
			plt.fill_between( np.linspace(min(time), max(time), len(time)*10), \
			     mu+std, \
			     mu-std, color=color, alpha=0.8, edgecolor="none", label = 'Best fit')


			plt.xlabel('BJD')
			plt.ylabel('Mag')
			plt.gca().invert_yaxis()
			plt.legend()
			plt.show()


			return best_params_passback, gp

		if emcee_lc==True:
			plt.scatter(time, mag, c='k', s= 10, alpha=0.5)
			plt.plot(time, lc_model.get_lc(time), 'g', alpha=0.8, label='Old')
			#plt.gca().invert_yaxis()
			plt.ylabel('Mag')
			plt.xlabel('Time')


			print('\n\n')
			print('~'*80)
			print("Posterior inference of lightcurve using emcee with GP")
			print('~'*80)
			print('\nRunning production chain') 

			initial = gp.get_parameter_vector()
			ndim, nwalkers = len(initial), 32
			sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (gp, mag, ellc_priors), threads = 8)

			print("Running burn-in...")
			p0 = initial + 1e-5 * np.random.randn(nwalkers, ndim)

		
			width=30
			start_time = time_.time()
			i, result=[], []
			for i, result in enumerate(sampler.sample(p0, iterations=emcee_draws)):
				n = int((width+1) * float(i) / emcee_draws)
				delta_t = time_.time()-start_time# time to do float(i) / n_steps % of caluculations
				time_incr = delta_t/(float(i+1) / emcee_draws) # seconds per increment
				time_left = time_incr*(1- float(i) / emcee_draws)
				m, s = divmod(time_left, 60)
				h, m = divmod(m, 60)
				sys.stdout.write('\r[{0}{1}] {2}% - {3}h:{4}m:{5:.2f}s'.format('#' * n, ' ' * (width - n), 100*float(i) / emcee_draws ,h, m, s))


			gp.compute(time, yerr = mag_err)
			mu, var = gp.predict(mag, np.linspace(min(time), max(time), \
			len(time)*10), return_var=True)
			std = np.sqrt(abs(var))
			color = "#ff7f0e"
			plt.fill_between( np.linspace(min(time), max(time), len(time)*10), \
			     mu+std, \
			     mu-std, color=color, alpha=0.8, edgecolor="none", label = 'Best fit')


			plt.xlabel('BJD')
			plt.ylabel('Mag')
			plt.gca().invert_yaxis()
			plt.legend()
			#plt.show()
			
			#names = gp.get_parameter_names()
			samples = sampler.chain[:, int(np.floor(emcee_draws*0.75)):, :].reshape((-1, ndim))
			corner.corner(samples, labels = gp.get_parameter_names())



			########################
			# Now save to fits file
			########################
			t = Table(sampler.flatchain, names=gp.get_parameter_names())
			t.add_column(Column(sampler.flatlnprobability,name='loglike'))
			indices = np.mgrid[0:nwalkers,0:emcee_draws]
			step = indices[1].flatten()
			walker = indices[0].flatten()
			t.add_column(Column(step,name='step'))
			t.add_column(Column(walker,name='walker'))
			try:
				t.write(emcee_chain_file)
			except:
				os.remove(emcee_chain_file)
				t.write(emcee_chain_file)

			lglike = t['loglike'][np.argmax(np.array(t['loglike']))]
			print('~'*80)
			print('Initial fit')
			print('~'*80)
			print('Loglike: {:.2f}'.format(lglike))
			print('Chi**2: {:.2f}'.format(-2*lglike))
			print('Reduced Chi**2: {:.2f}'.format(-2*lglike / (len(time) - len(gp.get_parameter_names()) ) ))
			print('BIC: {:.2f}'.format(len(gp.get_parameter_names())*np.log(len(time)) - 2*np.log(lglike)))
			print('~'*80)

			plt.show()
			return 



	else:

		# Initiate the model and get loglike
		lc_model = ellc_wrap()
		lc_model.set_param(ellc_params=ellc_params)

		if return_handler:
			return lc_model

		lglike = lc_model.get_log_like_lc(time, mag, mag_err, ellc_params, ellc_bounds, ellc_priors)

		if plot_model:
			if return_model:
				return lc_model.get_lc(time)
			else:
				# Make the plot
				plt.scatter(time, mag, c='k', s= 10, alpha=0.5)
				plt.plot(time, lc_model.get_lc(time), 'r', alpha=0.8)
				plt.gca().invert_yaxis()
				plt.ylabel('Mag')
				plt.xlabel('Time')

			
				# Return parameters where needed
				print('~'*80)
				print('Initial fit')
				print('~'*80)
				print('Loglike: {:.2f}'.format(lglike))
				print('Chi**2: {:.2f}'.format(-2*lglike))
				print('Reduced Chi**2: {:.2f}'.format(-2*lglike / (len(time) - len(free_params) ) ))
				print('BIC: {:.2f}'.format(len(free_params)*np.log(len(time)) - 2*np.log(-lglike)))
				print('~'*80)
				plt.show()

				return 

		if minimize_lc:
			# Output
			print('\n\n')
			print('~'*80)
			print("Minization using Scipy's L-BFGS-R algorithm")
			print('~'*80)
			print('\nRunning L-BFGS-s algorithm . . .', end='') 

			# Plot the initial data
			plt.scatter(time, mag, c='k', s= 10, alpha=0.5)
			plt.plot(time, lc_model.get_lc(time), 'g', alpha=0.8, label='Old')
			plt.gca().invert_yaxis()
			plt.ylabel('Mag')
			plt.xlabel('Time')


			# get param vector and bounds from free_params
			param_vector = [ellc_params[i] for i in free_params]
			bounds = [ellc_bounds[i] for i in free_params]

			# Cals scipy minimize with the L-BFGS_B algorithm
			soln = minimize(lc_model.scipy_minimize, param_vector, method="L-BFGS-B", bounds = bounds,  args=(time, mag, mag_err,free_params,ellc_bounds, ellc_priors),options={'disp': True})

			# Display output
			if soln.success:
				print(' Success in {} iterations'.format(soln.nit))
			else:
				print(' Failure in {} iterations'.format(soln.nit))
			print('Old loglike: {:.2f}'.format(lglike))
			print('new loglike: {:.2f}'.format(-soln.fun))
			print('Chi**2: {:.2f}'.format(-2*-soln.fun))
			print('Reduced Chi**2: {:.2f}'.format(-2*-soln.fun / (len(time) - len(free_params) ) ))
			print('BIC: {:.2f}'.format(len(free_params)*np.log(len(time)) - 2*np.log(2*soln.fun)))
			print('~'*70)
			print('{:>12}   {:>10}    {:>10}'.format('Parameter', 'Old', 'New'))
			for i in range(len(free_params)):
				print('{:>12}   {:>12.5f}    {:>12.5f}'.format(free_params[i], param_vector[i], soln.x[i] ))
			print('~'*70)


			# Create the dictionary with best params for pass back
			bett_params = ellc_params.copy()
			for i in range(len(free_params)):
				bett_params[free_params[i]] = soln.x[i]

			# Now set and plot the best model
			lc_model.set_param(ellc_params=bett_params)
			bett_params = lc_model.get_param()
			plt.plot(time, lc_model.get_lc(time), 'r', alpha=0.8, label='L-BFGS-R')
			plt.legend()
			plt.show()

			return bett_params

		if emcee_lc:
			# output
			print('\n\n')
			print('~'*80)
			print("Posterior inference of lightcurve using emcee")
			print('~'*80)
			print('\nRunning production chain') 

			# Plot data and initial model
			plt.scatter(time, mag, c='k', s= 10, alpha=0.5)
			plt.plot(time, lc_model.get_lc(time), 'g', alpha=0.8, label='Old')
			plt.gca().invert_yaxis()
			plt.ylabel('Mag')
			plt.xlabel('Time')

			# get param vector and bounds from free_params
			param_vector = [ellc_params[i] for i in free_params]
			ndim, nwalkers = len(param_vector), 4*len(param_vector)
		
			# initialse start position with finite loglike value
			p0 = []
			while len(p0) != nwalkers:
				p0_trial = np.random.normal(param_vector, 1e-6)
				lnlike_trial = lc_model.emcee_sampler(p0_trial, time, mag, mag_err,free_params, ellc_bounds, ellc_priors)
				if lnlike_trial!=-np.inf:
					p0.append(p0_trial)


			# Initiate the sampler
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lc_model.emcee_sampler, args = (time, mag, mag_err,free_params, ellc_bounds, ellc_priors), threads = 8)



			# run the sampler with custom progress bar
			width=30
			start_time = time_.time()
			i, result=[], []
			for i, result in enumerate(sampler.sample(p0, iterations=emcee_draws)):
				n = int((width+1) * float(i) / emcee_draws)
				delta_t = time_.time()-start_time# time to do float(i) / n_steps % of caluculations
				time_incr = delta_t/(float(i+1) / emcee_draws) # seconds per increment
				time_left = time_incr*(1- float(i) / emcee_draws)
				m, s = divmod(time_left, 60)
				h, m = divmod(m, 60)
				sys.stdout.write('\r[{0}{1}] {2:>5}% - {3}h:{4}m:{5:.2f}s'.format('#' * n, ' ' * (width - n), 100*float(i) / emcee_draws ,h, m, s))

		
			# Make the table			
			print('Making the table')
			t = Table(sampler.flatchain, names=free_params)

			# Valculate quadratic limb darkening values if needed
			if 'quad_ldc_1_1_' in free_params:
				quad_ldc_1_1, quad_ldc_1_2 = (np.array(t['quad_ldc_1_1_'])+ np.array(t['quad_ldc_1_2_']))**2,      0.5*np.array(t['quad_ldc_1_1_'])*(np.array(t['quad_ldc_1_1_']) + np.array(t['quad_ldc_1_2_']))**(-1)
				t.add_column(Column(quad_ldc_1_1,name='quad_ldc_1_1'))	
				t.add_column(Column(quad_ldc_1_2,name='quad_ldc_1_2'))	
			if 'quad_ldc_2_1_' in free_params:
				quad_ldc_2_1, quad_ldc_2_2 = (np.array(t['quad_ldc_2_1_'])+ np.array(t['quad_ldc_2_2_']))**2,      0.5*np.array(t['quad_ldc_2_1_'])*(np.array(t['quad_ldc_2_1_']) + np.array(t['quad_ldc_2_2_']))**(-1)		
				t.add_column(Column(quad_ldc_2_1,name='quad_ldc_2_1'))			
				t.add_column(Column(quad_ldc_2_2,name='quad_ldc_2_2'))					

			# add others
			t.add_column(Column(sampler.flatlnprobability,name='loglike'))
			indices = np.mgrid[0:nwalkers,0:emcee_draws]
			step = indices[1].flatten()
			walker = indices[0].flatten()
			t.add_column(Column(step,name='step'))
			t.add_column(Column(walker,name='walker'))
			try:
				t.write(emcee_chain_file)
			except:
				os.remove(emcee_chain_file)
				t.write(emcee_chain_file)

			# Find and plot the best solution
			print('Finding and plotting the best model')
			best_index = np.argmax(t['loglike'])
			bett_params = ellc_params.copy()
			best_params = np.array(list(t[best_index]))[:-3]
			for i in range(len(free_params)):
				bett_params[free_params[i]] = best_params[i]
			lc_model.set_param(ellc_params=bett_params)
			bett_params = lc_model.get_param()

			plt.plot(time, lc_model.get_lc(time), 'r', alpha=0.8, label='emcee')
			plt.legend()

			# Create a corner plot from the last 25% of sampler value
			samples = sampler.chain[:, int(np.floor(emcee_draws*0.75)):, :].reshape((-1, ndim))
			std_vals = np.std(samples, axis=0)
			means_vals = np.mean(samples, axis=0)
			corner_fig = corner.corner(samples, labels = free_params)

			

			# Output the best values
			print('~'*70)
			print('{:>12}   {:>10}    {:>10}'.format('Parameter', 'Old', 'New'))
			print('~'*70)
			for i in range(len(free_params)):
				print('{:>12}   {:>12.5f}    {:>12.5f}  +-  {:>12.5f}'.format(free_params[i], param_vector[i], means_vals[i],std_vals[i] ))
			print('~'*70)


			plt.show()
		
			return bett_params



		

