import easyastro
import numpy as np

time,rv,rv_err = np.loadtxt('AHMT.rv').T
time = time + 2400000

ellc_params, ellc_bounds, ellc_priors = easyastro.ellc_wrappers.get_ellc_all()

ellc_params['J'] = 2
ellc_params['V0'] = 4
ellc_bounds['t_zero'] = [ellc_params['t_zero']-0.1, ellc_params['t_zero']+0.1]
ellc_bounds['period'] = [2,3]
t = easyastro.fitting.fit_rv(time, rv, rv_err, ellc_params, ellc_bounds, ellc_priors, plot_model= False, minimize_rv = False, free_params=['K', 'f_c', 'f_s', 'V0', 'J'],
			emcee_rv=True, emcee_draws=100, emcee_chain_file='emcee_chain.fits')




