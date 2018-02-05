#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:59:24 2017

@author: sam
"""

#%%
# Imports
import numpy as np
from scipy.optimize import minimize
import celerite
from celerite import terms
import numpy as np
import matplotlib.pyplot as plt
from celerite.modeling import Model
import easyastro1 as ea

#%%
# First we are going to simulate a set of RV measurements taken at 
# random times over a 15 day time-span. We will assume a cadence of 30 mins,
# over 10 nights (assuming 5 nights are bad). An observing night lasts 7 hrs
# from the first day (time = 0.5). 

cadence = 1/(24*2)
def night_time(night, cadence):
    return np.arange(night-0.5 - 0.145, night-0.5 + 0.145, cadence)

time = np.array([ night_time(1,cadence),
                 night_time(2,cadence),
                 night_time(3,cadence),
                 night_time(5,cadence),
                 night_time(7,cadence),
                 night_time(8,cadence),
                 night_time(11,cadence),
                 night_time(12,cadence),
                 night_time(13,cadence),
                 night_time(15,cadence)]).flatten()
    
time_model = np.linspace(0,15,1000)

# Next we will simulate some RV data with given parameters
e = 0.2
w = 140 * np.pi / 180
fs = 0.1
fc = 0.1
K1 = 24.56
P = 0.6776
T0 = 3.54
V0 = 56.6
dV0 = 0

# So we can generate the TRUE radial velocity curve, and the model which we 
# will try to predict with GP

RV_TRUE, RV_not_used = ea.radial_velocity.calculate_radial_velocity(time, period = P,
                                                          T0 = T0, e = e, 
                                                          w = w, K1 = K1,
                                                          V0 = V0, dV0 = dV0)

RV_model, RV_not_used = ea.radial_velocity.calculate_radial_velocity(time_model, period = P,
                                                          T0 = T0, e = e, 
                                                          w = w, K1 = K1,
                                                          V0 = V0, dV0 = dV0)


RV_ERROR = np.random.uniform(2,5,RV_TRUE.shape[0])

plt.errorbar(time,RV_TRUE, yerr=RV_ERROR, fmt='ko')
plt.plot(time_model, RV_model, 'k', alpha=0.5)
plt.xlabel('Time [Day]')
plt.ylabel('RV [km/s]')
plt.grid()
plt.show()

#%%
# No we have an RV curve, lets ad a quasiperiodic signal caused by, lets
# say, the instrument.

# First lets perturb by the error
RV = np.random.normal(RV_TRUE, RV_ERROR)

# The lets add a high frequency signal at 4* the cadence
A_red_noise = 12.3
f =  1/(4*cadence)
RV = RV + A_red_noise*np.sin(2*np.pi*f)

plt.errorbar(time,RV, yerr=RV_ERROR, fmt='ko')
plt.xlabel('Time [Day]')
plt.ylabel('RV [km/s]')
plt.grid()
plt.show()


#%%
# OK, so now we have an RV curve with poisson noise and red noise. 
# Define the model
class _MeanModel(Model):
    parameter_names = ("K1", "fs", "fc", "P", "T0", "V0","dV0", "A_red_noise", "f_red_noise")

    def get_value(self, t):
        e = self.fs*self.fs + self.fc*self.fc
        w = np.arctan2(self.fs , self.fc)
        
        RV_model, RV_not_used = ea.radial_velocity.calculate_radial_velocity(t, period = self.P,
                                                                  T0 = self.T0, e = e, 
                                                                  w = w, K1 = self.K1,
                                                                  V0 = self.V0, dV0 = self.dV0)
        red_noise = self.A_red_noise * np.sin(2*np.pi*self.f_red_noise)

        return RV_model + red_noise
    
    
_bounds = dict( 
            K = (0,150), # K
           fs = (-0.99,0.99), # fs
           fc = (-0.99,0.99), # fc
         T_tr = (None, None), # T_tr
            P = (0.01, None), # P
           V0 = (None, None), # V0
          dV0 = (-1e-4, 1e-4), # DV0
  A_red_noise = (0.1, 20), # A_red
  f_red_noise = (0.1, 100)) # f_red
    




#%%
def fit_SB1_gp(time, rv, rv_err, emcee=False):
    ###########################################################################
    # The first step will be to iteratively plot starting params until a good
    # initila fit is found.

    # initilse default values 
    ###########################################################################
    fs = 0.0
    fc = 0.1
    K1 = 9
    T_tr = 0.0
    P = 1.0
    V0 = 2
    dV0=  0.0
    A_red_noise = 2
    f_red_noise = 0.2
    happy = 'n'
    
    e,w = 0,0
    
    ###################
    # Static parameters
    ###################
    # Start loop
    plt.ion()
    plt.show()
    while happy == 'n':
        plt.cla()
        
        #####################
        # Dynamic parameters
        #####################
        K1 = float(input('K1 [{}]:'.format(K1) ) or K1)
        e = float(input('e [{}]:'.format(e) ) or e)
        w = float(input('w [{}]:'.format(w) ) or w)
        fc = np.sqrt(e/(1+np.tan(w)**2))
        fs = e - fc**2 
        T_tr = float(input('T_tr [{}]:'.format(T_tr) ) or T_tr)
        P = float(input('P [{}]:'.format(P) ) or P)
        V0 = float(input('V0 [{}]:'.format(V0) ) or V0)
        dV0 = float(input('dV0 [{}]:'.format(dV0) ) or dV0)   
        

        mean_model = _MeanModel(K1=K1, fs=fs, fc=fc, P=P, T0=T_tr, V0=V0, dV0 = dV0, A_red_noise=A_red_noise, f_red_noise=f_red_noise, bounds = _bounds)
        
        plt.errorbar(time,rv, yerr=rv_err, fmt='ko')
        plt.plot(np.linspace(min(time), max(time), 1000), mean_model.get_value(np.linspace(min(time), max(time), 1000)), 'r')
        plt.xlabel('Time [d]')
        plt.ylabel('RV [km/s]')
        plt.grid()
        plt.draw()
        plt.pause(0.001)
    
        happy = input("Are you happy (y/n)? [n] : ") or 'n'



    kernel = terms.RealTerm(log_a=3.0937, log_c=-2.302)
    gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
    
    gp.compute(time, rv_err)
    print("\n\nLoglikliehoods\n~~~~~~~~~~~~~~\nInitial log-likelihood: {:.3f}".format(gp.log_likelihood(rv)))
    #return gp


    # Make the maximum likelihood prediction
    time_model = np.linspace(min(time), max(time), 1000)
    gp.compute(time_model, yerr=np.interp(time_model, time, rv_err))
    mu, var = gp.predict(np.interp(time_model, time, rv), t=time_model, return_var=True)
    std = np.sqrt(abs(var))
    color = "#ff7f0e"
    plt.close()
    plt.errorbar(time, rv, yerr=rv_err, fmt=".k", capsize=0)
    plt.plot(time_model, mu, color=color)
    plt.fill_between(time_model, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
    plt.ylabel(r"$RV [km/s]$")
    plt.xlabel(r"$Time [d]$")
    plt.title("maximum likelihood prediction");
    plt.grid()
    

    
    # Define a cost function
    def neg_log_like(params, rv, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(rv)
    
    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    
    
    
    gp.compute(time, rv_err)

    soln = minimize(neg_log_like, initial_params, #jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(rv, gp),
                    options={'gtol': 1e-6, 'disp': True})
    print("  Final log-likelihood: {:.3f}".format(-soln.fun))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for i in range(len(gp.get_parameter_names())):
        print('{}:        {:.3f}'.format(gp.get_parameter_names()[i], soln.x[i]))


    if emcee:
        def log_probability(params):
            gp.set_parameter_vector(params)
            lp = gp.log_prior()
            if not np.isfinite(lp):
                return -np.inf
    
            try:
                return gp.log_likelihood(time) + lp
            except:
                return -np.inf
            
        import emcee
    
        initial = np.array(soln.x)
        ndim, nwalkers = len(initial), 32
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        
        print("Running burn-in...")
        p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 500)
        
        import corner
        names = gp.get_parameter_names()
        cols = mean_model.get_parameter_names()
        inds = np.array([names.index("mean:"+k) for k in cols])
        
        
        corner.corner(sampler.flatchain[:, inds], truths=initial,
                      labels=cols);

    