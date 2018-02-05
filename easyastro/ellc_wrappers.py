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

def get_ellc_params():
	return dict(radius_1=0.1,
		k = 0.05,
		zp = 0.0,
		sbratio=0.0, 
		b = 0.05, 
		light_3=0.0, 
		t_zero=2458439.675, 
		period=2.199188, 
		a=1.0, 
		K=25,
		q=0.2, 
		f_c=0.0, 
		f_s=0.0,
		V0 = 3.0,
		dV0 = 0.0, 
		J=2,

		lin_ldc_1=0.6, 
		lin_ldc_2=0.6,

		quad_ldc_1_1 =  0.46,
		quad_ldc_1_2 =  0.15,
		quad_ldc_2_1 =  0.46,
		quad_ldc_2_2 =  0.15,
		quad_ldc_1_1_ =  (0.46 + 0.15)**2,
		quad_ldc_1_2_ =  0.5*0.46*(0.46 + 0.15)**(-1),
		quad_ldc_2_1_ =  (0.46 + 0.15)**2,
		quad_ldc_2_2_ =  0.5*0.46*(0.46 + 0.15)**(-1),

		claret_ldc_1_1 =  0.58,
		claret_ldc_1_2 =  0.58,
		claret_ldc_1_3 =  0.58,
		claret_ldc_1_4 =  0.58,
		claret_ldc_2_1 =  0.58,
		claret_ldc_2_2 =  0.58,
		claret_ldc_2_3 =  0.58,
		claret_ldc_2_4 =  0.58,

		gdc_1=0.0, 
		gdc_2=0.0,
		T_ld_1 = 5777,
		M_ld_1 = 0.0,
		L_ld_1 = 4.44, 
		T_ld_2 = 5777,
		M_ld_2 = 0.0,
		L_ld_2 = 4.44, 
		didt=0.0, 
		domdt=0.0, 
		rotfac_1=1.0, 
		rotfac_2=1.0, 
		bfac_1=None, 
		bfac_2=None, 
		heat_1=None, 
		heat_2=None, 
		lambda_1=0.0, 
		lambda_2=0.0, 
		vsini_1=0.0, 
		vsini_2=0.0, 
		t_exp=None, 
		n_int=None, 
		grid_1='default', 
		grid_2='default', 
		ld_1='claret', 
		ld_2='lin', 
		shape_1='sphere', 
		shape_2='sphere', 
		spots_1=None, 
		spots_2=None, 
		exact_grav=False, 
		band='V',
		flux_weighted=False,
		verbose=1)







def get_ellc_bounds():
	return dict(radius_1=[0.1,0.6],
		k = [0.001,0.9],
		zp = [-np.inf, np.inf],
		sbratio=[0,1], 
		b = [0,1], 
		light_3=[0,1], 
		t_zero=[-np.inf, np.inf], 
		period=[0.01,100], 
		a=[0,500], 
		K=[1,50],
		q=[0,1],
		f_c=[-0.5,0.5],
		f_s=[-0.5,0.5],
		V0 = [-50,50],
		dV0 = [-1e-5,1e-5], 
		J = [0.1,5],

		lin_ldc_1=[0,1], 
		lin_ldc_2=[0,1],

		quad_ldc_1_1 =  [0.4,0.7],
		quad_ldc_1_2 = [0.05,0.3],
		quad_ldc_2_1 =  [0.4,0.7],
		quad_ldc_2_2 =  [0.05,0.3],

		quad_ldc_1_1_ =  [0.2,0.5],
		quad_ldc_1_2_ =  [0.2,0.5],
		quad_ldc_2_1_ =  [0.2,0.5],
		quad_ldc_2_2_ =  [0.2,0.5],	

		claret_ldc_1_1 =  [0,1],
		claret_ldc_1_2 =  [0,1],	
		claret_ldc_1_3 =  [-1,1],	
		claret_ldc_1_4 =  [-1,1],	
		claret_ldc_2_1 =  [0,1],	
		claret_ldc_2_2 =  [0,1],	
		claret_ldc_2_3 =  [-1,1],	
		claret_ldc_2_4 =  [-1,1],		
		gdc_1=[0,1], 
		gdc_2=[0,1],
		T_ld_1 = [4000,8000],
		M_ld_1 = [-1.,1],
		L_ld_1 = [3.5,5],
		T_ld_2 = [4000,8000],
		M_ld_2 = [-1, 1],
		L_ld_2 = [3.5,5], 
		lambda_1=[0,360], 
		lambda_2=[0,360],
		vsini_1=[0,500], 
		vsini_2=[0,500])




def get_ellc_priors(ellc_params=None):
	if ellc_params!=None:
		return dict(t_zero = [ellc_params['t_zero'], 1e-5],
			    period = [ellc_params['period'], 1e-5],
			    T_ld_1 = [ellc_params['T_ld_1'], 124])
	else:
		return dict()

def get_ellc_prior_values(ellc_params, ellc_bounds, ellc_priors, specific_val=None):
	#print(ellc_params)
	param_keys = np.array([i for i in ellc_params.keys()])
	bound_keys = np.array([i for i in ellc_bounds.keys()])
	prior_keys = np.array([i for i in ellc_priors.keys()])
	params_with_bounds = list(set( [i for i in ellc_params.keys()]).intersection( [i for i in ellc_bounds.keys()]))

	#print(params_with_bounds)

	if specific_val==None:		
		prior = int(0)
		for i in params_with_bounds:
			low, val, high = ellc_bounds[i][0], ellc_params[i],  ellc_bounds[i][1]
			#print(i, low, val, high)
			if (val<low) or (val>high):
				#print('FAIL', i, low, val, high)
				return np.inf
			else:
				if i in prior_keys:
					prior += (ellc_params[i] - ellc_priors[i][0])**2 / (ellc_priors[i][1])**2


		return prior
	
	else:
		low, val, high = ellc_bounds[specific_val][0], ellc_params[specific_val],  ellc_bounds[specific_val][1]
		if (val<low) or (val>high):
			#print('FAIL', specific_val,  low, val, high)
			return np.inf
		else:
			if specific_val in prior_keys:
				return (ellc_params[specific_val] - ellc_priors[specific_val][0])**2 / (ellc_priors[specific_val][1])**2
			else:
				return int(0)	

def get_ellc_all():
	return 	get_ellc_params(), get_ellc_bounds(), get_ellc_priors()


			
def check_ellc_input(ellc_params,ellc_bounds, ellc_priors):
	print('~'*80)
	print('Parameters with bounds')
	print('~'*80)
	print('{:>20}   {:>10}    {:>10}    {:>10}   {:>10}    {:>10}'.format('Parameter', 'Low', 'Value', 'High', 'Within?' , 'Prior chi'))
	print('~'*80)
	params_with_bounds = list(set( [i for i in ellc_params.keys()]).intersection( [i for i in ellc_bounds.keys()]))
	params_without_bounds = np.setdiff1d(np.array([i for i in ellc_params.keys()]), np.array([i for i in ellc_bounds.keys()]))


	for i in params_with_bounds:
		low,val,high = ellc_bounds[i][0], np.round(ellc_params[i],5), ellc_bounds[i][1]
		are_in='No'
		if (val>=low) and (val<=high):
			are_in='Yes'
		prior = get_ellc_prior_values(ellc_params, ellc_bounds, ellc_priors, specific_val=i)

		print('{:>20}   {:>10}    {:>10}    {:>10}   {:>10}    {:>10}'.format(i, low, val, high, are_in, np.round(prior,2)))
	print('~'*80)
	print('Parameters without bounds')
	print('~'*80)
	for i in params_without_bounds:
		print('{:>12}   {:>10}'.format(i, str(ellc_params[i])))
	print('~'*80)
	print('~'*80)
	


class ellc_wrap():
	parameter_names = ('radius_1','k','zp','sbratio', 'b', 'light_3', 't_zero', 'period', 'a', 'K','q', 'f_c', 'f_s','V0', 'dV0', 
			'lin_ldc_1','lin_ldc_2' , 
			'quad_ldc_1_1', 'quad_ldc_1_2','quad_ldc_2_1','quad_ldc_2_2', 'quad_ldc_1_1_', 'quad_ldc_1_2_','quad_ldc_2_1_','quad_ldc_2_2_', 
			'claret_ldc_1_1','claret_ldc_1_2','claret_ldc_1_3','claret_ldc_1_4','claret_ldc_2_1','claret_ldc_2_2','claret_ldc_2_3','claret_ldc_2_4', 
			'gdc_1', 'gdc_2', 
			'T_ld_1', 'M_ld_1', 'L_ld_1','T_ld_2', 'M_ld_2', 'L_ld_2', 
			'didt','domdt', 'rotfac_1', 'rotfac_2','bfac_1','bfac_2', 'heat_1','heat_2', 
			'lambda_1','lambda_2', 'vsini_1','vsini_2', 
			't_exp','n_int', 'grid_1', 'grid_2', 'ld_1' , 'ld_2', 'shape_1', 'shape_2', 'spots_1', 'spots_2', 'exact_grav', 
			'band', 'flux_weighted','verbose')

	def return_all_params():
		return np.array(['radius_1','k','zp','sbratio', 'b', 'light_3', 't_zero', 'period', 'a', 'K','q', 'f_c', 'f_s','V0', 'dV0',  
			'lin_ldc_1','lin_ldc_2' , 
			'quad_ldc_1_1', 'quad_ldc_1_2','quad_ldc_2_1','quad_ldc_2_2', 'quad_ldc_1_1_', 'quad_ldc_1_2_','quad_ldc_2_1_','quad_ldc_2_2_', 
			'claret_ldc_1_1','claret_ldc_1_2','claret_ldc_1_3','claret_ldc_1_4','claret_ldc_2_1','claret_ldc_2_2','claret_ldc_2_3','claret_ldc_2_4', 
			'gdc_1', 'gdc_2', 
			'T_ld_1', 'M_ld_1', 'L_ld_1','T_ld_2', 'M_ld_2', 'L_ld_2', 
			'didt','domdt', 'rotfac_1', 'rotfac_2','bfac_1','bfac_2', 'heat_1','heat_2', 
			'lambda_1','lambda_2', 'vsini_1','vsini_2', 
			't_exp','n_int', 'grid_1', 'grid_2', 'ld_1' , 'ld_2', 'shape_1', 'shape_2', 'spots_1', 'spots_2', 'exact_grav', 
			'band', 'flux_weighted','verbose'])
	
	def get_value(self, t):
		chi_prior = self.get_ellc_prior_values(ellc_params, ellc_bounds, ellc_priors)
		if np.isinf(chi_prior):
			return np.array([np.inf for i in range(len(t))])
		else:
			return self.get_lc(self, t)

	def get_lc(self, t):
		############################################
		# Sort out limb darkening coeffs for star 1
		############################################
		if (self.ld_1=='lin'):
			ldc_coeffs1 = self.lin_ldc_1
		elif (self.ld_1=='quad'):
			ldc_coeffs1 = [self.quad_ldc_1_1, self.quad_ldc_1_2]
		elif (self.ld_1=='claret'):
			ldc_coeffs1 = [self.claret_ldc_1_1, self.claret_ldc_1_2, self.claret_ldc_1_3, self.claret_ldc_1_4]

		############################################
		# Sort out limb darkening coeffs for star 2
		############################################
		if (self.ld_2=='lin'):
			ldc_coeffs2 = self.lin_ldc_2
		elif (self.ld_2=='quad'):
			ldc_coeffs2 = [self.quad_ldc_2_1, self.quad_ldc_2_2]
		elif (self.ld_2=='claret'):
			ldc_coeffs2 = [self.claret_ldc_2_1, self.claret_ldc_2_2, self.claret_ldc_2_3, self.claret_ldc_2_4]


		########################
		# Make the call to ellc
		########################
		try:
			return self.zp - np.log10(ellc.lc(t, radius_1 = self.radius_1 , radius_2 = self.radius_2, sbratio = self.sbratio, incl = self.incl, light_3=self.light_3,
				 t_zero=self.t_zero, period=self.period, a=self.a, q=self.q, f_c=self.f_c, f_s=self.f_s, ldc_1=ldc_coeffs1,
				 ldc_2=ldc_coeffs2, gdc_1=self.gdc_1, gdc_2=self.gdc_2, didt=self.didt, domdt=self.domdt, rotfac_1=self.rotfac_1,
				 rotfac_2=self.rotfac_2, bfac_1=self.bfac_1, bfac_2=self.bfac_2, heat_1=self.heat_1, heat_2=self.heat_2, lambda_1=self.lambda_1,
				 lambda_2=self.lambda_2, vsini_1=self.vsini_1, vsini_2=self.vsini_2, t_exp=self.t_exp, n_int=self.n_int, grid_1=self.grid_1,
				grid_2=self.grid_2, ld_1=self.ld_1, ld_2=self.ld_2, shape_1=self.shape_1, shape_2=self.shape_2, spots_1=self.spots_1,
				 spots_2=self.spots_2, exact_grav=self.exact_grav, verbose=self.verbose))
		except:
			return np.array([np.inf for i in range(len(t))])





	def get_rv(self, t):
		############################################
		# Sort out limb darkening coeffs for star 1
		############################################
		if (self.ld_1=='lin'):
			ldc_coeffs1 = self.lin_ldc_1
		elif (self.ld_1=='quad'):
			ldc_coeffs1 = [self.quad_ldc_1_1, self.quad_ldc_1_2]
		elif (self.ld_1=='claret'):
			ldc_coeffs1 = [self.claret_ldc_1_1, self.claret_ldc_1_2, self.claret_ldc_1_3, self.claret_ldc_1_4]

		############################################
		# Sort out limb darkening coeffs for star 2
		############################################
		if (self.ld_2=='lin'):
			ldc_coeffs2 = self.lin_ldc_2
		elif (self.ld_2=='quad'):
			ldc_coeffs2 = [self.quad_ldc_2_1, self.quad_ldc_2_2]
		elif (self.ld_2=='claret'):
			ldc_coeffs2 = [self.claret_ldc_2_1, self.claret_ldc_2_2, self.claret_ldc_2_3, self.claret_ldc_2_4]



		if self.flux_weighted==False:
			#print('bad')
			return self.V0 + self.dV0*(self.t_zero - t) +  np.array(ellc.rv(t, radius_1 = self.radius_1 , radius_2 = self.radius_2, sbratio = self.sbratio, incl = self.incl,
				 t_zero=self.t_zero, period=self.period, a=self.a, q=self.q, f_c=self.f_c, f_s=self.f_s, ldc_1=self.lin_ldc_1,
				 ldc_2=self.lin_ldc_2, gdc_1=self.gdc_1, gdc_2=self.gdc_2, didt=self.didt, domdt=self.domdt, rotfac_1=self.rotfac_1,
				 rotfac_2=self.rotfac_2, bfac_1=self.bfac_1, bfac_2=self.bfac_2, heat_1=self.heat_1, heat_2=self.heat_2, lambda_1=self.lambda_1,
				 lambda_2=self.lambda_2, vsini_1=self.vsini_1, vsini_2=self.vsini_2, t_exp=self.t_exp, n_int=self.n_int, grid_1=self.grid_1,
				grid_2=self.grid_2, ld_1=self.ld_1, ld_2=self.ld_2, shape_1=self.shape_1, shape_2=self.shape_2, spots_1=self.spots_1,
				 spots_2=self.spots_2, flux_weighted=self.flux_weighted, verbose=self.verbose))

		elif self.flux_weighted==True:
			#print('Here')
			return self.V0 + self.dV0*(self.t_zero - t) +  np.array(ellc.rv(t,self.radius_1 , self.radius_2, self.sbratio, self.incl, q=self.q, 
											a=self.a, lambda_1=self.lambda_1, vsini_1=self.vsini_1,
                     									ld_1=self.ld_1,ldc_1=ldc_coeffs1,  
                     									grid_1='default',grid_2='default',
                     									flux_weighted=True,f_c=self.f_c,f_s=self.f_s))	
		else:
			print('What')								





	def calculate_log_like_prior(self, ellc_params, ellc_bounds, ellc_priors, specific_val = None):
		param_keys = np.array([i for i in ellc_params.keys()])
		bound_keys = np.array([i for i in ellc_bounds.keys()])
		prior_keys = np.array([i for i in ellc_priors.keys()])
		params_with_bounds = list(set( [i for i in ellc_params.keys()]).intersection( [i for i in ellc_bounds.keys()]))

		if specific_val==None:          
			prior = int(0)
			for i in params_with_bounds:
				low, val, high = ellc_bounds[i][0], ellc_params[i],  ellc_bounds[i][1]
				if (val<low) or (val>high):
					#print('PROBLEM', i, low, val, high)
					return -np.inf
				else:
					if i in prior_keys:
				        	prior += (ellc_params[i] - ellc_priors[i][0])**2 / (ellc_priors[i][1])**2


			return -0.5*prior

		else:
			low, val, high = ellc_bounds[specific_val][0], ellc_params[specific_val],  ellc_bounds[specific_val][1]
			if (val<low) or (val>high):
				#print('PROBLEM', i, low, val, high)
				return np.inf
			else:
				if specific_val in prior_keys:
					return (ellc_params[specific_val] - ellc_priors[specific_val][0])**2 / (ellc_priors[specific_val][1])**2
				else:
					return int(0)   


	def get_rv_log_like(self, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors):
		# First set parameters
		#self.set_param(ellc_params)
		# The calculate prior and return  -np.inf if out
		lp = self.calculate_log_like_prior(self.get_param(), ellc_bounds, ellc_priors)
		if lp == -np.inf:
			return np.inf
		
		# Now caluclate model
		rv_model = self.get_rv(time)[0]

		# Now calculate loglike
		wt = 1.0/(rv_err**2 + self.J**2)
		loglike = -0.5* np.sum(( (rv-rv_model)**2 * wt - np.log(wt))) + lp
		
		return -loglike


	def get_log_like_rv(self,time, rv, rv_err, ellc_params, ellc_bounds, ellc_priors):
		lp = self.calculate_log_like_prior(self.get_param(), ellc_bounds, ellc_priors, specific_val = None)
		#print('lp: {}'.format(lp))
		if lp==-np.inf:
			return -np.inf

		return -self.get_rv_log_like(time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors)
	
	def get_rv_chi(self, params,param_names, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors):
		return -2*self.get_rv_log_like(params,param_names, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors)

	def scipy_minimize_rv(self,params,param_names, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors):
		# Set the relevent parameters

		tmp = dict()
		for i in range(len(param_names)):
			exec("tmp['{}'] = {}".format(param_names[i], params[i]))
		
		# now set	
		self.set_param(tmp)
		# Now return the loglike
		return self.get_rv_log_like(time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors)

	def emcee_rv_log_like(self,params,param_names, time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors):
		# Set the relevent parameters

		tmp = dict()
		for i in range(len(param_names)):
			exec("tmp['{}'] = {}".format(param_names[i], params[i]))
		
		# now set	
		self.set_param(tmp)
		# Now return the loglike
		return -self.get_rv_log_like(time,rv,rv_err, ellc_params, ellc_bounds, ellc_priors)

	def set_parameter(self, name, value):
		exec("self.{} = {}".format(name, value))

	def set_param(self, ellc_params):
		##########################################
		# Set the parameters via an exec command
		##########################################
		for i in ellc_params:
			exec("self.{} = ellc_params['{}']".format(i,i))

		##########################################
		# Conversions and calculations
		##########################################
		self.radius_2 = self.radius_1 * self.k
		self.incl = 180*np.arccos(self.b*self.radius_1)/np.pi
		self.e = self.f_c**2 + self.f_s**2
		if self.e>0.9:
			self.e=0.9
			
		#print(self.K, self.e, self.f_c, self.f_s, self.q, self.period, self.incl)
		if self.K != 0:
			self.a = 0.019771142*self.K*np.sqrt(1-self.e**2)*(1+1/self.q)*self.period /np.sin(np.pi*self.incl/180)

		#############################################
		# Calculate limb-darkening values if needed
		#############################################
		# For liner, we do not need to do anything
		#
		#
		############################################################################
		# For quadratic, we use the nomencleture from Kipping (2013MNRAS.435.2152K)
		# We convert the quad_ldc_*_*_ parameters to the real quad_ldc_*_* parameters
		# for both stars which will be used in ellc - bounds apply only to the latter 
		# calculated values.
		############################################################################
		if self.ld_1 == 'quad':
			#print('quad')
			self.quad_ldc_1_1, self.quad_ldc_1_2 = 2*np.sqrt(self.quad_ldc_1_1_)*self.quad_ldc_1_2_, np.sqrt(self.quad_ldc_1_1_)*(1-2*self.quad_ldc_1_2_)
		if self.ld_2 == 'quad':
			self.quad_ldc_2_1, self.quad_ldc_2_2 = 2*np.sqrt(self.quad_ldc_2_1_)*self.quad_ldc_2_2_, np.sqrt(self.quad_ldc_2_1_)*(1-2*self.quad_ldc_2_2_)


		###################################################################
		# For claret, we will use the coeffieint lookup provided with ellc.
		# To do this, we will take T_ld_*, L_ld_* and M_ld_* to lookup all
		# four coefficients and the gravity darkening coeficient. 
		###################################################################
		if self.ld_1 == 'claret':
			lda = ellc.ldy.LimbGravityDarkeningCoeffs(self.band)
			c1,c2,c3,c4,g1 = lda(self.T_ld_1, self.L_ld_1, self.M_ld_1)
			if np.isnan(c1):
				print('LDC 1 failure   {}    {}    {}'.format(self.T_ld_1, self.L_ld_1, self.M_ld_1))
			self.claret_ldc_1_1, self.claret_ldc_1_2, self.claret_ldc_1_3, self.claret_ldc_1_4, self.gdc_1 = c1,c2,c3,c4,g1

		if self.ld_2 == 'claret':
			lda = ellc.ldy.LimbGravityDarkeningCoeffs(self.band)
			c1,c2,c3,c4,g1 = lda(self.T_ld_2, self.L_ld_2, self.M_ld_2)
			if np.isnan(c1):
				print('LDC 2 failure   {}    {}    {}'.format(self.T_ld_1, self.L_ld_1, self.M_ld_1))
			self.claret_ldc_2_1, self.claret_ldc_2_2, self.claret_ldc_2_3, self.claret_ldc_2_4,self.gdc_2 = c1,c2,c3,c4,g1


		####################################
		# Now catch if gdc_* have type None
		####################################
		if self.gdc_1 is None:
			self.gdc_1 = 0.0
		if self.gdc_2 is None:
			self.gdc_2 - 0.0


	def get_param(self):
		return dict(radius_1=self.radius_1,
			k = self.k,
			zp = self.zp,
			sbratio=self.sbratio, 
			b = self.b, 
			light_3=self.light_3, 
			t_zero=self.t_zero, 
			period=self.period, 
			a=self.a, 
			K=self.K,
			q=self.q, 
			f_c=self.f_c, 
			f_s=self.f_s,
			V0 = self.V0,
			dV0 = self.dV0, 
			J = self.J,
			lin_ldc_1=self.lin_ldc_1, 
			lin_ldc_2=self.lin_ldc_1,

			quad_ldc_1_1 =  self.quad_ldc_1_1,
			quad_ldc_1_2 =  self.quad_ldc_1_2,
			quad_ldc_2_1 =  self.quad_ldc_2_1,
			quad_ldc_2_2 =  self.quad_ldc_2_2,
			quad_ldc_1_1_ = self.quad_ldc_1_1_,
			quad_ldc_1_2_ = self.quad_ldc_1_2_,
			quad_ldc_2_1_ = self.quad_ldc_2_1_,
			quad_ldc_2_2_ = self.quad_ldc_2_2_,

			claret_ldc_1_1 =  self.claret_ldc_1_1,
			claret_ldc_1_2 =  self.claret_ldc_1_2,
			claret_ldc_1_3 =  self.claret_ldc_1_3,
			claret_ldc_1_4 =  self.claret_ldc_1_4,
			claret_ldc_2_1 =  self.claret_ldc_2_1,
			claret_ldc_2_2 =  self.claret_ldc_2_2,
			claret_ldc_2_3 =  self.claret_ldc_2_3,
			claret_ldc_2_4 =  self.claret_ldc_2_4,
	
			gdc_1=self.gdc_1, 
			gdc_2=self.gdc_2,

			T_ld_1 = self.T_ld_1,
			M_ld_1 = self.M_ld_1,
			L_ld_1 = self.L_ld_1, 
			T_ld_2 = self.T_ld_2,
			M_ld_2 = self.M_ld_2,
			L_ld_2 = self.L_ld_2, 
			didt=self.didt, 
			domdt=self.domdt, 
			rotfac_1=self.rotfac_1, 
			rotfac_2=self.rotfac_2,
			bfac_1=self.bfac_1, 
			bfac_2=self.bfac_2, 
			heat_1=self.heat_1, 
			heat_2=self.heat_2, 
			lambda_1=self.lambda_1, 
			lambda_2=self.lambda_2, 
			vsini_1=self.vsini_1, 
			vsini_2=self.vsini_2,
			t_exp=self.t_exp, 
			n_int=self.n_int,
			grid_1=self.grid_1,
			grid_2=self.grid_2,
			ld_1=self.ld_1, 
			ld_2=self.ld_2, 
			shape_1=self.shape_1, 
			shape_2=self.shape_2, 
			spots_1=self.spots_1,
			spots_2=self.spots_2,
			exact_grav=self.exact_grav, 
			band=self.band,
			flux_weighted=self.flux_weighted,
			verbose=self.verbose)


	def get_log_like_lc(self, time, mag, mag_err, ellc_params, ellc_bounds, ellc_priors):
		# set params
		self.set_param(ellc_params)

		# get prior
		chi_prior = get_ellc_prior_values(ellc_params, ellc_bounds, ellc_priors)

		if chi_prior==np.inf:
			return -np.inf

		# calculate model
		mag_model = self.get_lc(time)
		if (True in np.isnan(mag_model)):
			return -np.inf

		# calculate chi
		chi_data = (mag - mag_model)**2 / (mag_err**2)

		# return loglike
		return np.sum(-0.5*(chi_prior + chi_data))

	




	def scipy_minimize(self, theta, time, mag, mag_err,free_params,ellc_bounds, ellc_priors):
		# set params
		tmp = dict()
		for i in range(len(theta)):
			tmp[free_params[i]] = theta[i]
		self.set_param(tmp)

		return -2* self.get_log_like_lc(time, mag, mag_err,tmp , ellc_bounds, ellc_priors)




	def emcee_sampler(self, theta, time, mag, mag_err,free_params, ellc_bounds, ellc_priors):
		# set params
		tmp = dict()
		for i in range(len(theta)):
			tmp[free_params[i]] = theta[i]
		self.set_param(tmp)



		# get prior
		chi_prior = get_ellc_prior_values(self.get_param(), ellc_bounds, ellc_priors)

	
		if chi_prior==np.inf:
			return -np.inf

		# calculate model
		try:
			mag_model = self.get_lc(time)
			if (True in np.isnan(mag_model)):
				return -np.inf

			if False in np.isfinite(mag_model):
				return -np.inf

		except:
			return -np.inf

		# calculate chi
		chi_data = (mag - mag_model)**2 / (mag_err**2)

		# return loglike
		return np.sum(-0.5*(chi_prior + chi_data))






class ellc_GP_wrap(Model):
	'''
	def __init__(self,  ellc_GP_params, not_included_params, ellc_gp_bounds ):

		# dynamically unpack
		#self.parameter_names =tuple(ellc_GP_params.keys())
		for i in ellc_GP_params:
			exec("self.{} = 1".format(i))
	
		#self.parameter_boudns = ellc_gp_bounds

		#		('radius_1', 'k', 'zp', 'sbratio', 'b', 'light_3', 't_zero', 'period', 'a', 'K', 'q', 'f_c', 'f_s', 'V0', 'dV0', \
		#		'lin_ldc_1', 'lin_ldc_2', 'quad_ldc_1_1', 'quad_ldc_1_2', 'quad_ldc_2_1', 'quad_ldc_2_2', 'quad_ldc_1_1_', 'quad_ldc_1_2_', 'quad_ldc_2_1_', 'quad_ldc_2_2_', \
		#		'claret_ldc_1_1', 'claret_ldc_1_2', 'claret_ldc_1_3', 'claret_ldc_1_4', 'claret_ldc_2_1', 'claret_ldc_2_2', 'claret_ldc_2_3', 'claret_ldc_2_4', 'gdc_1', 'gdc_2', \
		#		 'T_ld_1', 'M_ld_1', 'L_ld_1', 'T_ld_2', 'M_ld_2', 'L_ld_2', 'didt', 'domdt', 'rotfac_1', 'rotfac_2', 'lambda_1', 'lambda_2', 'vsini_1', 'vsini_2', 'verbose')
	'''
	parameter_names = ('radius_1',
			 'k',
			 'zp',
			 'sbratio',
			 'b',
			 'light_3',
			 't_zero',
			 'period',
			 'a',
			 'K',
			 'q',
			 'f_c',
			 'f_s',
			 'V0',
			 'dV0',
			 'lin_ldc_1',
			 'lin_ldc_2',
			 'quad_ldc_1_1',
			 'quad_ldc_1_2',
			 'quad_ldc_2_1',
			 'quad_ldc_2_2',
			 'quad_ldc_1_1_',
			 'quad_ldc_1_2_',
			 'quad_ldc_2_1_',
			 'quad_ldc_2_2_',
			 'claret_ldc_1_1',
			 'claret_ldc_1_2',
			 'claret_ldc_1_3',
			 'claret_ldc_1_4',
			 'claret_ldc_2_1',
			 'claret_ldc_2_2',
			 'claret_ldc_2_3',
			 'claret_ldc_2_4',
			 'gdc_1',
			 'gdc_2',
			 'T_ld_1',
			 'M_ld_1',
			 'L_ld_1',
			 'T_ld_2',
			 'M_ld_2',
			 'L_ld_2',
			 'didt',
			 'domdt',
			 'rotfac_1',
			 'rotfac_2',
			 'lambda_1',
			 'lambda_2',
			 'vsini_1',
			 'vsini_2',
			 'verbose',
			 'J')

	def get_lc(self, t):
		##########################################
		# Conversions and calculations
		##########################################
		radius_2 = self.radius_1 * self.k
		incl = 180*np.arccos(self.b*self.radius_1)/np.pi
		e = self.f_c**2 + self.f_s**2

		if self.K != 0:
			a = 0.019771142*self.K*np.sqrt(1-e**2)*(1+1/self.q)*self.period /np.sin(np.pi*incl/180)

		#############################################
		# Calculate limb-darkening values if needed
		#############################################
		# For liner, we do not need to do anything
		#
		#
		############################################################################
		# For quadratic, we use the nomencleture from Kipping (2013MNRAS.435.2152K)
		# We convert the quad_ldc_*_*_ parameters to the real quad_ldc_*_* parameters
		# for both stars which will be used in ellc - bounds apply only to the latter 
		# calculated values.
		############################################################################
		if self.not_included_params['ld_1'] == 'quad':
			ldc_coeffs1 = [2*np.sqrt(self.quad_ldc_1_1_)*self.quad_ldc_1_2_, np.sqrt(self.quad_ldc_1_1_)*(1-2*self.quad_ldc_1_2_)]
		if self.not_included_params['ld_2'] == 'quad':
			ldc_coeffs2 = [2*np.sqrt(self.quad_ldc_2_1_)*self.quad_ldc_2_2_, np.sqrt(self.quad_ldc_2_1_)*(1-2*self.quad_ldc_2_2_)]


		###################################################################
		# For claret, we will use the coeffieint lookup provided with ellc.
		# To do this, we will take T_ld_*, L_ld_* and M_ld_* to lookup all
		# four coefficients and the gravity darkening coeficient. 
		###################################################################
		if self.not_included_params['ld_1'] == 'claret':
			lda = ellc.ldy.LimbGravityDarkeningCoeffs(self.not_included_params['band'])
			c1,c2,c3,c4,g1 = lda(self.T_ld_1, self.L_ld_1, self.M_ld_1)
			if np.isnan(c1):
				print('LDC 1 failure   {}    {}    {}'.format(self.T_ld_1, self.L_ld_1, self.M_ld_1))
			ldc_coeffs1, gdc_1 = [c1,c2,c3,c4],g1

		if self.not_included_params['ld_2'] == 'claret':
			lda = ellc.ldy.LimbGravityDarkeningCoeffs(self.not_included_params['band'])
			c1,c2,c3,c4,g1 = lda(self.T_ld_2, self.L_ld_2, self.M_ld_2)
			if np.isnan(c1):
				print('LDC 2 failure   {}    {}    {}'.format(self.T_ld_1, self.L_ld_1, self.M_ld_1))
			ldc_coeffs2, gdc_2 = [c1,c2,c3,c4],g1


		############################################
		# Sort out limb darkening coeffs for star 1
		############################################
		if (self.not_included_params['ld_1']=='lin'):
			ldc_coeffs1 = self.lin_ldc_1
		if (self.not_included_params['ld_2']=='lin'):
			ldc_coeffs2 = self.lin_ldc_1

		########################
		# Make the call to ellc
		########################
		return self.zp - np.log10(ellc.lc(t, radius_1 = self.radius_1 , radius_2 = radius_2, sbratio = self.sbratio, incl = incl, light_3=self.light_3,
			 t_zero=self.t_zero, period=self.period, a=self.a, q=self.q, f_c=self.f_c, f_s=self.f_s, ldc_1=ldc_coeffs1,
			 ldc_2=ldc_coeffs2, gdc_1=self.gdc_1, gdc_2=self.gdc_2, didt=self.didt, domdt=self.domdt, rotfac_1=self.rotfac_1,
			 rotfac_2=self.rotfac_2, bfac_1=self.not_included_params['bfac_1'], bfac_2=self.not_included_params['bfac_2'], heat_1=self.not_included_params['heat_1'], heat_2=self.not_included_params['heat_2'], lambda_1=self.lambda_1,
			 lambda_2=self.lambda_2, vsini_1=self.vsini_1, vsini_2=self.vsini_2, t_exp=self.not_included_params['t_exp'], n_int=self.not_included_params['n_int'], grid_1=self.not_included_params['grid_1'],
			grid_2=self.not_included_params['grid_2'], ld_1=self.not_included_params['ld_1'], ld_2=self.not_included_params['ld_2'], shape_1=self.not_included_params['shape_1'], 
			shape_2=self.not_included_params['shape_2'], spots_1=self.not_included_params['spots_1'], 
			 spots_2=self.not_included_params['spots_2'], exact_grav=self.not_included_params['exact_grav'], verbose=self.verbose))


	def get_value(self, t ):
		for i in self.not_included_params:
			exec("self.{} = self.not_included_params['{}']".format(i,i)) # this should set 
		chi_prior = get_ellc_prior_values(self.ellc_GP_params, self.ellc_gp_bounds, self.ellc_gp_priors)
		if np.isinf(chi_prior):
			return np.array([np.inf for i in range(len(t))])
		else:
			return self.get_lc(t)

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)
		




def get_ellc_GP_params(ellc_params, ellc_bounds, free_params):
	ellc_GP_params=dict()
	not_included_params=dict()
	ellc_gp_bounds = dict()
	for i in ellc_params:
		if (isinstance(ellc_params[i], int) or isinstance(ellc_params[i], float)) and (type(ellc_params[i]) != bool):
			exec("ellc_GP_params['{}'] = ellc_params['{}']".format(i, i))
			if i in free_params:
				exec("ellc_gp_bounds['{}'] = ellc_bounds['{}']".format(i, i))
		else:
			exec("not_included_params['{}'] = ellc_params['{}']".format(i, i))

	return ellc_GP_params,not_included_params,ellc_gp_bounds


def log_probability(params, gp, mag, param_priors):
	# set the GP params
	try:
		#print('h')
		gp.set_parameter_vector(params)

		# Check to see if its out of bounds
		lp = gp.log_prior()
		if not np.isfinite(lp):
			return -np.inf

		# get _priors loglike
		log_like_priors = prior_log_likes(gp, param_priors)

		return  gp.log_likelihood(mag) + log_like_priors
	except:
		return -np.inf


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



def phaser(x, y,y_err, t_zero, period):
	x_phase = (((x-t_zero)/period + 0.5) % 1) -0.5
	zipped_sorted = sorted(zip(x_phase, y,y_err ))
	x_phase_sorted = np.array([i[0] for i in zipped_sorted])
	y_sorted = np.array([i[1] for i in zipped_sorted])
	y_err_sorted = np.array([i[2] for i in zipped_sorted])
	return x_phase_sorted, y_sorted,y_err_sorted




















