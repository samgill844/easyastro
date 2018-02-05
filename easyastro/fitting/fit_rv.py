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
from easyastro.ellc_wrappers import get_ellc_params, get_ellc_bounds, get_ellc_priors, get_ellc_all, check_ellc_input, ellc_wrap, phaser
from scipy.optimize import fmin_l_bfgs_b

def fit_rv(time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors, plot_model = False, minimize_rv=True, emcee_rv=True, emcee_draws=500, emcee_chain_file='emcee_chain.fits', free_params=['K', 'f_c', 'f_s', 'V0'],return_handler=False):

	# Initate and set the parameters
	rv_model = ellc_wrap()
	rv_model.set_param(ellc_params=ellc_params)

	if return_handler:
		return rv_model

	# maske a copy of the dictionary, and set t_zero and period to 0.0 and 1.0 respectivvely
	ellc_params_phase = ellc_params.copy()
	ellc_params_phase['t_zero']=0.0
	ellc_params_phase['period']=1.0
	phase_ = np.linspace(-0.5,0.5,1000)


	if plot_model:
		# sort and plot model
		time_phase, rv_sorted,rv_err_err_sorted = phaser(time,rv,rv_err, ellc_params['t_zero'], ellc_params['period'])
		plt.errorbar(time_phase, rv_sorted, yerr = rv_err_err_sorted, fmt='ko')

		# set phase dict to rv_model
		rv_model.set_param(ellc_params=ellc_params_phase)	
		plt.plot(phase_, rv_model.get_rv(phase_)[0], 'r', alpha=0.8)

		# Format the graph
		plt.ylabel('RV [km/s]')
		plt.xlabel('phase')
		plt.grid()
		
		# Print the desired output
		print('~'*80)
		print('Initial fit')
		print('~'*80)
		print('Chi: {}'.format(-2*rv_model.emcee_rv_log_like([ellc_params['K']],['K'], time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors) ))
		print('Loglike: {}'.format(rv_model.get_log_like_rv(time, rv, rv_err, ellc_params, ellc_bounds, ellc_priors)))
		print('~'*80)

		plt.show()


	if minimize_rv:
		# Obtain the parameter values of free_params along with its bounds
		initial_params = [ellc_params[i] for i in free_params]
		better_params = np.copy(initial_params)
		bounds_ = [ellc_bounds[i] for i in free_params]

		# Estimate a step size based on 1000th of its range
		step_size = [ (bounds_[i][1] - bounds_[i][0])/1000 for i in range(len(bounds_))]
		if True in np.isinf(step_size):
			raise ValueError('Step size is inf. Check the bounds on the free_params - It needs finite values!!')

	
		# Now call scipy's minimze with the L-BFGS-B algorithm
		soln = minimize(rv_model.scipy_minimize_rv, initial_params, method="L-BFGS-B", args=(free_params, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors), \
				tol=None, callback=None, options={'disp': True,'epsilon' : 1}, bounds = bounds_)
		print('-'*10)

	
		# Now print the best params whilst creating a optimized dictionary that we
		# will pass back
		better_dict = ellc_params.copy()
		for i in range(len(soln.x)):
			print('{}:  {}'.format(free_params[i],soln.x[i] ))
			exec("better_dict['{}'] = {}".format(free_params[i] , soln.x[i]))

		# sort and plot model
		time_phase, rv_sorted,rv_err_err_sorted = phaser(time,rv,rv_err, better_dict['t_zero'], better_dict['period'])
		plt.errorbar(time_phase, rv_sorted, yerr = rv_err_err_sorted, fmt='go')

		# set phase dict to rv_model
		better_dict_phase = better_dict.copy()
		better_dict_phase['t_zero'] = 0.0
		better_dict_phase['period'] = 1.0
		rv_model.set_param(ellc_params=better_dict_phase)	
		plt.plot(phase_, rv_model.get_rv(phase_)[0], 'r-', alpha=0.8, label='Optimized')


		# set phase dict to rv_model
		better_dict_phase = ellc_params.copy()
		better_dict_phase['t_zero'] = 0.0
		better_dict_phase['period'] = 1.0
		rv_model.set_param(ellc_params=better_dict_phase)	
		plt.plot(phase_, rv_model.get_rv(phase_)[0], 'r', alpha=0.2)

		# Get
		plt.ylabel('RV [km/s]')
		plt.xlabel('phase')
		plt.grid()
		plt.show()

		return better_dict

	if emcee_rv:
		print('\n\n')
		print('~'*80)
		print("Posterior inference of SB1 using emcee")
		print('~'*80)
		print('\nRunning production chain\n') 

		# Obtain the parameter values of free_params along with its bounds
		initial_params = [ellc_params[i] for i in free_params]
		better_params = ellc_params.copy()
		bounds_ = [ellc_bounds[i] for i in free_params]
		step_size = [ (bounds_[i][1] - bounds_[i][0])/1000 for i in range(len(bounds_))] # make sure bounds is not inf!
		if True in np.isinf(step_size):
			raise ValueError('Step size is inf. Check the bounds on the free_params - It needs finite values!!')

		# Get ndim and nwalkers from free_params
		ndim = len(free_params)
		nwalkers = 4*ndim

	
		# initiate starting postions with finite loglikelihood values
		p0 = []
		while len(p0) < nwalkers:
			p0_trial = np.random.normal(initial_params, step_size)
	
			if rv_model.emcee_rv_log_like(p0_trial,free_params, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors) != -np.inf:
				p0.append(p0_trial)
	
		

		# Create the sampler
		sampler = emcee.EnsembleSampler(nwalkers, ndim, rv_model.emcee_rv_log_like, args = (free_params, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors), threads = 8)
			

		# Call the sample with a custom progress bar
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


		# Now get the last 25% of draws and make a corner plot
		samples = sampler.chain[:, int(np.floor(emcee_draws*0.75)):, :].reshape((-1, ndim))
		corner.corner(samples, labels = free_params)
		std_vals = np.std(samples, axis=0)

		# Now save to fits file
		t = Table(sampler.flatchain, names=free_params)
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


		# Display the step with the highest loglike and create the best parameter dictionary
		best_index = np.argmax(t['loglike'])
		best_params = np.array(list(t[best_index]))[:-3]
		print('\n\n')
		print('~'*80)
		print('Best parameters with in chain {} step {} with loglike {:.3f}'.format(t['step'][best_index],t['walker'][best_index], t['loglike'][best_index] ))
		print('~'*80)
		for i in range(len(best_params)):
			print("{:>10} = {:>10.5} +- {:>10.5}".format(free_params[i], best_params[i], std_vals[i] ))
			exec("better_params['{}'] = {}".format(free_params[i], best_params[i]))
		print('~'*80)


		# Now set the best step
		better_params_ = better_params.copy()
		better_params_['t_zero'] = 0.0
		better_params_['period'] = 1.0

		# sort and plot model
		fig2 = plt.figure()
		rv_model.set_param(better_params_)
		time_phase, rv_sorted,rv_err_err_sorted = phaser(time,rv,rv_err, ellc_params['t_zero'], ellc_params['period'])
		plt.errorbar(time_phase, rv_sorted, yerr = rv_err_err_sorted, fmt='ko')
		plt.plot(phase_, rv_model.get_rv(phase_)[0], 'r', alpha=0.8)

		# format
		plt.ylabel('RV [km/s]')
		plt.xlabel('phase')
		plt.grid()
		plt.show()
		

		return better_params

			

