import numpy as np
import emcee, glob
import sys,os, time as time_
import corner
import corner2
from astropy.table import Table, Column
import matplotlib.pyplot as plt
import matplotlib


def create_starting_positions(function, starting_parameters, error = 1e-4, nwalkers=100): 
	p0 = []
	while len(p0)<nwalkers:
		trial_pos = np.random.normal(starting_parameters, error)
		if function(trial_pos) != -np.inf:
			p0.append(trial_pos)
	return p0

def run_sampler(sampler, p0, emcee_draws, width):
	start_time = time_.time()
	i, result=[], []
	for i, result in enumerate(sampler.sample(p0, iterations=emcee_draws)):
		n = int((width+1) * float(i) / emcee_draws)
		delta_t = time_.time()-start_time# time to do float(i) / n_steps % of caluculations
		time_incr = delta_t/(float(i+1) / emcee_draws) # seconds per increment
		time_left = time_incr*(1- float(i) / emcee_draws)
		m, s = divmod(time_left, 60)
		h, m = divmod(m, 60)
		sys.stdout.write('\r[{0}{1}] {2:.2f}% - {3}h:{4}m:{5:.2f}s'.format('#' * n, ' ' * (width - n), 100*float(i) / emcee_draws ,h, m, s))

	return sampler

def print_and_return_last_quarter_of_sampler(sampler, emcee_draws, nwalkers,  parameter_names, print_vals=True,  emcee_chain_file = None):
	# get samples
	samples = sampler.chain[:, int(np.floor(emcee_draws*0.75)):, :].reshape((-1, len(parameter_names)))
	std_vals = np.std(samples, axis=0)
	means_vals = np.mean(samples, axis=0)
	median_vals = np.median(samples, axis=0)


	# get corner
	fig = corner.corner(samples, labels = parameter_names, truths = median_vals)


	# print parameter
	if print_vals:
		print('~'*70)
		print('{:>12}   {:>10}'.format('Parameter', 'Value'))
		print('~'*70)
		for i in range(len(parameter_names)):
			print('{:>12}   {:>12.5f}  +-  {:>12.5f}'.format(parameter_names[i], median_vals[i], std_vals[i] ))
		print('~'*70)

	# Now create table
	t = Table(sampler.flatchain, names=parameter_names)
	t.add_column(Column(sampler.flatlnprobability,name='loglike'))
	indices = np.mgrid[0:nwalkers,0:emcee_draws]
	step = indices[1].flatten()
	walker = indices[0].flatten()
	t.add_column(Column(step,name='step'))
	t.add_column(Column(walker,name='walker'))
	if emcee_chain_file != None:
		try:
		    t.write(emcee_chain_file)
		except:
		    os.remove(emcee_chain_file)
		    t.write(emcee_chain_file)

	return samples, means_vals, median_vals, std_vals, fig, t

def make_corner_from_table(filename, labels = None, remove_cols=[], font_size=10,coner_2=False):

	matplotlib.rcParams.update({'font.size': font_size})
	t = Table.read(filename)


	best_index = np.argmax(np.array(t['loglike']))
	best_par = list(t[best_index])[:-3]
	print('Best model:')
	print('\tLoglike: {}'.format(t['loglike'][best_index]))
	print('\tChi: {}'.format(-0.5*t['loglike'][best_index]))
	print('\tWalker: {}'.format(t['walker'][best_index]))
	print('\tStep: {}'.format(t['step'][best_index]))
	samples = int(np.max(t['step'])*0.75)
	mask = (t['step'] < samples)

	t.remove_columns(['loglike', 'walker', 'step'])
	for i in remove_cols:
		t.remove_column(i)

	samples = t[~mask].as_array().view(np.float64).reshape(t[~mask].as_array().shape + (-1,))
	medians = np.median(samples, axis=0)
	tt=[]
	for i in range(medians.shape[0]):
		tt.append([best_par[i], medians[i]])

	print("Parameters")
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	for i in range(medians.shape[0]):
		print("{:>10} {:>25.5f} {:>25.5f}    [{:>10.5f} , {:>10.2f} % ]".format(t.colnames[i], tt[i][0], tt[i][1], tt[i][0] - tt[i][1], 100* (tt[i][0] - tt[i][1])/ tt[i][0] ))

	if coner_2==True:
		if labels==None:
			fig = corner2.corner(samples, labels = t.colnames, truths = tt)
		else:
			fig = corner2.corner(samples, labels = labels, truths = tt)

		return fig, tt, t.colnames
	if coner_2==False:
		if labels==None:
			fig = corner.corner(samples, labels = t.colnames)
		else:
			fig = corner.corner(samples, labels = labels)

		return fig, tt, t.colnames

def compare_fits_files_for_parameter(list_of_fits_files=glob.glob('joint_fit*.fits'), parameter='k', labels = None):
	
	colours = ['k', 'b', 'r', 'y']
	fig1 = plt.figure()

	# load and hist parameter
	for i in range(len(list_of_fits_files)):
		t = Table.read(list_of_fits_files[i])
		
		best_index = np.argmax(np.array(t['loglike']))
		best_val = t[parameter][best_index]

		samples = int(np.max(t['step'])*0.75)
		mask = (t['step'] < samples)

		vals = t[parameter][~mask]
		median = np.median(vals, axis=0)
		
		if labels==None:
			label = list_of_fits_files[i][:-5]
		else:
			label = labels[i]

		plt.hist(vals, 100, color=colours[i], label=label, alpha = 1.0/len(list_of_fits_files), normed=1)
		plt.axvline(median, c=colours[i], ls = '--')
		plt.axvline(best_val, c=colours[i])
	
	
	plt.xlabel(parameter)
	plt.ylabel('Normalised density')
	plt.legend()

	return fig1














	
