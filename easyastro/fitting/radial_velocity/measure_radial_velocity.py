import waveletspec as w
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def _sampling_uniform_in_velocity(wave_base, wave_top, velocity_step):
    """
    Create a uniformly spaced grid in terms of velocity:

    - An increment in position (i => i+1) supposes a constant velocity increment (velocity_step).
    - An increment in position (i => i+1) does not implies a constant wavelength increment.
    - It is uniform in log(wave) since:
          Wobs = Wrest * (1 + Vr/c)^[1,2,3..]
          log10(Wobs) = log10(Wrest) + [1,2,3..] * log10(1 + Vr/c)
      The last term is constant when dealing with wavelenght in log10.
    - Useful for building the cross correlate function used for determining the radial velocity of a star.
    """
    # Speed of light
    c = 299792.4580 # km/s
    #c = 299792458.0 # m/s

    ### Numpy optimized:
    # number of elements to go from wave_base to wave_top in increments of velocity_step
    i = int(np.ceil( (c * (wave_top - wave_base)) / (wave_base*velocity_step)))
    grid = wave_base * np.power((1 + (velocity_step / c)), np.arange(i)+1)

    # Ensure wavelength limits since the "number of elements i" tends to be overestimated
    wfilter = grid <= wave_top
    grid = grid[wfilter]

    ### Non optimized:
    #grid = []
    #next_wave = wave_base
    #while next_wave <= wave_top:
        #grid.append(next_wave)
        ### Newtonian version:
        #next_wave = next_wave + next_wave * ((velocity_step) / c) # nm
        ### Relativistic version:
        ##next_wave = next_wave + next_wave * (1.-np.sqrt((1.-(velocity_step*1000.)/c)/(1.+(velocity_step*1000.)/c)))

    return np.asarray(grid)



def find_rv(spectrum, use_flux_err=False, model_spectra_params = [5777, 0.0, 4.44, 2.2], velocity_step=0.5, velocity_limits = [-200, 200], resolution=55000 ) :

	# 1. resample spectra in velocity space
	waveobs_v = _sampling_uniform_in_velocity(np.min(spectrum['waveobs']), np.max(spectrum['waveobs']), velocity_step)
	flux = np.interp(waveobs_v, spectrum['waveobs'], spectrum['flux'], left=0.0, right=0.0)
	err = np.interp(waveobs_v, spectrum['waveobs'], spectrum['err'], left=0.0, right=0.0)

	# 2. Now resample model spectra
	model = w.interpolate_spectra(Teff=model_spectra_params[0], MH=model_spectra_params[1], Logg=model_spectra_params[2], Vsini=model_spectra_params[0], resolution=resolution)
	model = w.resample(model, np.min(spectrum['waveobs']), np.max(spectrum['waveobs']), len(spectrum['waveobs']))
	model_flux =  np.interp(waveobs_v, model['waveobs'], model['flux'], left=0.0, right=0.0)

	# 4. Subtract the means otherwise correlation is bad
	model_flux -= np.mean(model_flux)
	flux -= np.mean(flux)

	# 5. cross correlate using fft convolve
	corr_img =  fftconvolve(model_flux, flux[::-1], mode='same')
	best_velocity = (np.argmax(corr_img) - corr_img.shape[0]/2) * velocity_step * -1
	
	#plt.plot( ( np.arange(corr_img.shape[0]) - corr_img.shape[0]/2  )  * velocity_step * -1 , corr_img )
	plt.show()


	print('Best RV: {:.2f} km/s'.format(best_velocity))
	

	return waveobs_v, waveobs_vv, flux,model_flux ,corr_img



