#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:55:11 2017

@author: sam
"""

import numpy as np
from scipy.optimize import minimize
import celerite
from celerite import terms
import numpy as np
import matplotlib.pyplot as plt
from celerite.modeling import Model
import ellc, emcee, corner
from astropy.table import Table, Column
import time as time_
import sys, os
#%%


class _SB1_Model(Model):
    parameter_names = ('r1', 'k', 'sbratio', 'b', 'q', 'fs', 'fc', 'K', 'T_tr', 'P', 'Pdot', 'T_ld', 'M_ld', 'Logg_ld', 'V0', 'dV0')

    def get_value(self, t, flux_weighted=False):
        # Get r2
        r2 = self.r1*self.k
        incl = 180*np.arccos(self.b*self.r1)/np.pi 
       
        # get limb darkening coeffs
        a1,a2,a3,a4,g1 = ellc.ldy.LimbGravityDarkeningCoeffs('V')(self.T_ld, self.Logg_ld, self.M_ld)
        ldc = [a1,a2,a3,a4]
        
        # Correct for drift
        tau = t - self.Pdot*(t-self.T_tr)

            
        # Now do preliminaries
        e = self.fc**2 + self.fs**2
        a = 0.019771142*self.K*np.sqrt(1-e**2)*(1+1/self.q)*self.P /np.sin(np.pi*incl/180)
    
    
        # Then make the call     
        rv1,rv2 = ellc.rv(t, radius_1=self.r1, radius_2=r2, sbratio=self.sbratio,
                          incl=incl, q=self.q, a=a, ld_1='claret',ldc_1=ldc,ld_2='lin',
                          f_s = self.fs, f_c = self.fc,
                          ldc_2=0.45, t_zero=self.T_tr, period=self.P, flux_weighted=flux_weighted)
        
        
        # now add in drift in velocity
        rv1 = rv1 + self.V0 +  self.dV0*(t - self.T_tr)



		
        
        return rv1


def get_sb1_initial():
	return dict(r1 = 0.2,
			k = 0.5,
			sbratio = 0.2,
			b = 0.05,
			q = 0.5,
			fs = 0.0,
			fc = 0.0,
			K = 100,
			T_tr = 0.0,
			P = 1.5,
			Pdot = 0,
			T_ld = 5777,
			M_ld = 0.0,
			Logg_ld = 4.44,
			V0 = 0.0,
			dV0 = 0.0)


def get_sb1_bounds():
	return dict(r1 = (0,0.9),
			k = (0, 0.8),
			sbratio = (-0.00001, 1),
			b = (0, 1),
			q = (0, 1),
			fs = (-1,1),
			fc = (-1,1),
			K = (0,400),
			T_tr = (-np.inf, np.inf),
			P = (-np.inf, np.inf),
			Pdot = (-1e-5, 1e-5),
			T_ld = (4000,8000),
			M_ld = (-1,1),
			Logg_ld = (3.5,5),
			V0 = (-100,100),
			dV0 = (-1e-5, 1e-5))



def get_sb1_priors(initial_params=None):
    if initial_params != None:
        return dict( T_ld = [initial_params['T_ld'], 124],
                    P = [initial_params['P'], 1e-5],
                    T_tr = [initial_params['T_tr'], 1e-4])                
    else:
        return dict()


def check_sb1_input(params, bounds, priors):
    print('~'*80)
    print('{:>12}   {:>10}    {:>10}    {:>10}   {:>10}    {:>10}'.format('Parameter', 'Low', 'Value', 'High', 'Within?' , 'Prior chi'))
    print('~'*80)
    for i in list(params.keys()):
	    low, high = bounds[i]
	    val = params[i]
	    yes = 'No'
	    if (val>low) and (val< high):
		    yes = 'Yes'
		
	    prior = 0.0
	    #print(list(priors.keys()))
	    #print(i)
	    if i in list(priors.keys()):
	        prior = 0.5*(params[i] - priors[i][0])**2 / priors[i][1]**2
	        
	    print('{:>12}   {:>10}    {:>10}    {:>10}   {:>10}    {:>10}'.format(i, str(low), str(val), str(high), yes, np.round(prior,2)))


def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)
    
    
def prior_log_likes(gp, param_priors):
    log_like_val = 0.0
    for key in list(param_priors.keys()):
        prior, prior_err = param_priors[key]
        #print(prior, prior_err)
        if 'mean:{}'.format(key) in gp.parameter_names:
            idx = np.where(np.array(gp.parameter_names)=='mean:{}'.format(key))
            param_val = gp.parameter_vector[idx][0]
            log_like_val += -0.5*(param_val - prior)**2 / prior_err**2
            
        elif 'kernel:{}'.format(key) in gp.parameter_names:
            idx = np.where(np.array(gp.parameter_names)=='kernel:{}'.format(key))
            param_val = gp.parameter_vector[idx][0]
            log_like_val += -0.5*(param_val - prior)**2 / prior_err**2
            
    return  log_like_val 



def prior_log_likes(gp, param_priors):
    log_like_val = 0.0
    for key in list(param_priors.keys()):
        prior, prior_err = param_priors[key]
        #print(prior, prior_err)
        if 'mean:{}'.format(key) in gp.parameter_names:
            idx = np.where(np.array(gp.parameter_names)=='mean:{}'.format(key))
            param_val = gp.parameter_vector[idx][0]
            log_like_val += -0.5*(param_val - prior)**2 / prior_err**2
            
        elif 'kernel:{}'.format(key) in gp.parameter_names:
            idx = np.where(np.array(gp.parameter_names)=='kernel:{}'.format(key))
            param_val = gp.parameter_vector[idx][0]
            log_like_val += -0.5*(param_val - prior)**2 / prior_err**2
            
    return  log_like_val  

def log_probability(params, gp, mag, param_priors,flux_weighted):
    # set the GP params
    gp.set_parameter_vector(params)
    
    # Check to see if its out of bounds
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf

    # get _priors loglike
    log_like_priors = prior_log_likes(gp, param_priors)
   
    try:
        return  gp.log_likelihood(mag) + log_like_priors
    except:
        return -np.inf



def fit_SB1_gp(time, rv,rv_err, initial_params, bounds, priors, flux_weighted=False, plot_guess=True,
		freeze_parameters=['r1', 'k', 'sbratio', 'b', 'q', \
		 'Pdot', 'T_ld', 'M_ld',\
		 'Logg_ld', 'dV0'], emcee_fit=False, draws =1000):
	print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print('RV fitting procedure with Gaussian Processes')
	print('S. Gill (s.gill@keele.ac.uk)')
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
	mean_model = _SB1_Model(**initial_params, bounds = bounds)
	
	time_upsampled = np.linspace(min(time), max(time), len(time)*1000)
	if plot_guess:
		print('Plotting initial Guess. . .')
		plt.plot(time_upsampled,mean_model.get_value(time_upsampled), 'r--', linewidth=2, \
			label='Starting guess', alpha  = 0.7)

		plt.errorbar(time,rv,yerr=rv_err, fmt='ko')
		plt.legend()
		plt.show()
		return

	######################
	# Initiate the kernel
	######################
	#kernel = (terms.RealTerm(log_a=-9.77, log_c=1.91) )
	kernel = (terms.Matern32Term(log_sigma=-2, log_rho = 2))

	########################
	# Initiate the GP model
	########################
	gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)

	########################################
	# Freeze parameter we do ot wish to fit
	########################################
	for i in freeze_parameters:
		gp.freeze_parameter('mean:{}'.format(i))	


	#############################################
	# Compute and calculate the initial log like
	#############################################
	gp.compute(time, rv_err)
	
	print('Minimisation of lightcurve with GP')
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print("{:>20}: {:.3f}".format('Initial log-like',gp.log_likelihood(rv)))

	##################
	# Now optimise
	##################
	initial_params = gp.get_parameter_vector()
	bounds_ = gp.get_parameter_bounds()

	soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds_, args=(rv, gp))

	###################
	# Print the output
	###################
	print("{:>20}: {:.3f} ({} iterations)".format('Final log-like', -soln.fun, soln.nit))
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	#for i in range(len(gp.get_parameter_names())):
	#	print('{:.3f}             {}'.format(soln.x[i], gp.get_parameter_names()[i]))

	############################################################################
	# Now get the best model as a fictionary so we can put it back in and plot
	############################################################################
	best_params = {}
	for i in range(len(gp.parameter_names)):
		if gp.parameter_names[i][:5]=='mean:':
			best_params[gp.parameter_names[i][5:]] = gp.parameter_vector[i]
			print('{:>12}:      {:.3f}'.format(gp.parameter_names[i][5:], gp.parameter_vector[i]))

	#########################
	# Now get the best model
	#########################
	plt.cla()

	plt.errorbar(time,rv, yerr=rv_err,fmt = 'ko', alpha = 0.3)

	# Plot the best model (just to be sure)
	mean_model_better = _SB1_Model(**best_params, bounds=bounds)
	plt.plot(time,mean_model_better.get_value(time), 'b--', linewidth=2, \
		 label='Model selected')

	# Plot the variance with upsampled time series
	mu, var = gp.predict(rv, time_upsampled, return_var=True)
	std = np.sqrt(abs(var))
	color = "#ff7f0e"
	plt.fill_between( time_upsampled, \
		 mu+std, \
		 mu-std, color=color, alpha=0.8, edgecolor="none", label = 'Best fit')


	plt.xlabel('Time')
	plt.ylabel('RV [km/s]')
	plt.legend()
	plt.show()



	if emcee_fit:
		initial = np.array(soln.x)
		ndim, nwalkers = len(initial), 32
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (gp, rv, priors, flux_weighted), threads = 8)

		print("Running burn-in...")
		p0 = initial + 1e-5 * np.random.randn(nwalkers, ndim)

		burn_in = draws
		width=30
		start_time = time_.time()
		i, result=[], []
		for i, result in enumerate(sampler.sample(p0, iterations=burn_in)):
			n = int((width+1) * float(i) / burn_in)
			delta_t = time_.time()-start_time# time to do float(i) / n_steps % of caluculations
			time_incr = delta_t/(float(i+1) / burn_in) # seconds per increment
			time_left = time_incr*(1- float(i) / burn_in)
			m, s = divmod(time_left, 60)
			h, m = divmod(m, 60)
			sys.stdout.write('\r[{0}{1}] {2}% - {3}h:{4}m:{5:.2f}s'.format('#' * n, ' ' * (width - n), 100*float(i) / burn_in ,h, m, s))

		names = gp.get_parameter_names()
		cols = mean_model.get_parameter_names()
		inds = np.array([names.index("mean:"+k) for k in cols])
		samples = sampler.chain[:, int(np.floor(draws*0.75)):, :].reshape((-1, ndim))
		print(samples.shape, cols)
		corner.corner(samples, labels = gp.get_parameter_names())
		plt.show()

		########################
		# Now save to fits file
		########################
		from astropy.table import Table, Column
		chain_file = 'rv_fit.fits'
		try:
			os.remove(chain_file)
		except:
			pass

		t = Table(sampler.flatchain, names=gp.get_parameter_names())
		t.add_column(Column(sampler.flatlnprobability,name='loglike'))
		indices = np.mgrid[0:nwalkers,0:burn_in]
		step = indices[1].flatten()
		walker = indices[0].flatten()
		t.add_column(Column(step,name='step'))
		t.add_column(Column(walker,name='walker'))
		t.write(chain_file)




       
