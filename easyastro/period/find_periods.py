import numba as nb
import numpy as np
import emcee, corner
import time as time_, sys, os
from astropy.table import Table, Column


def lnlike(theta, t, m ,me):
	a, b, w = theta
	
	if (a < 0) or (b < 0) or (w < 0):
		return -np.inf

	if (w > 100):
		return -np.inf

	model = a*np.sin(t*w) + b*np.cos(t*w)

	if me==None:
		me = m

	return -0.5 * np.sum( ((m**2 / me**2) - (m**2 - model**2)/me**2 ) / (m**2 / me**2)   )

	


def period_finder(t,m,me=None, emcee_draws=1000, emcee_chain_file='period_find.fits'):

	ndim, nwalkers = 3, 100

	p0 = [ [np.random.uniform(0.1,0.2), np.random.uniform(0.1,0.2), np.random.uniform(0.1,0.2)] for i in range(nwalkers)]

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args = (t, m ,me))

	width=30
	start_time = time_.time()
	for i, result in enumerate(sampler.sample(p0, iterations=emcee_draws)):
		n = int((width+1) * float(i) / emcee_draws)
		delta_t = time_.time()-start_time# time to do float(i) / n_steps % of caluculations
		time_incr = delta_t/(float(i+1) / emcee_draws) # seconds per increment
		time_left = time_incr*(1- float(i) / emcee_draws)
		m, s = divmod(time_left, 60)
		h, m = divmod(m, 60)
		sys.stdout.write('\r[{0}{1}] {2}% - {3}h:{4}m:{5:.2f}s'.format('#' * n, ' ' * (width - n), 100*float(i) / emcee_draws ,h, m, s))


	#names = gp.get_parameter_names()
	samples = sampler.chain[:, int(np.floor(emcee_draws*0.75)):, :].reshape((-1, ndim))
	fig = corner.corner(samples, labels = [r'$a$',r'$b$', r'$\omega$'])



	########################
	# Now save to fits file
	########################
	t = Table(sampler.flatchain, names=['a', 'b', 'w'])
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

	return fig, t
