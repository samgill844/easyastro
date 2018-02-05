from astropy.io import fits
from heapq import nsmallest
import numpy as np
import matplotlib.pyplot as plt

h = fits.open('Kurkucks_grid.fits')


def create_spectrum_structure(waveobs, flux=None, err=None):
    """
    Create spectrum structure
    """
    spectrum = np.recarray((len(waveobs), ), dtype=[('waveobs', float),('flux', float),('err', float)])
    spectrum['waveobs'] = waveobs

    if flux is not None:
        spectrum['flux'] = flux
    else:
        spectrum['flux'] = 0.0

    if err is not None:
        spectrum['err'] = err
    else:
        spectrum['err'] = 0.0

    return spectrum


def interpolate_Kurkuckz(Teff = 5400, logg=4.3):
	T = [i[0] for i in h[1].data[1:]]
	L = [i[1] for i in h[1].data[1:]]
	Tu = np.unique(T)
	Lu = np.unique(L)

	# find closest 2
	closest_teff = nsmallest(2, Tu, key=lambda x: abs(x-Teff))
	closest_teff.sort()
	closest_logg = nsmallest(2, Lu, key=lambda x: abs(x-logg))
	closest_logg.sort()

	# create models indexes
	indexs=[]
	for i in closest_teff:
		for j in closest_logg:
			print(i, min(Tu), Tu[1], Tu[0])
			print( int( (i - min(Tu) ) / (Tu[1] - Tu[0]) )  )
			indexs.append( int( (i - min(Tu) ) / (Tu[1] - Tu[0]) )*7 + int( (j - min(Lu)) / (Lu[1] - Lu[0]) )+ 1)

	xd = (Teff - closest_teff[0])/(closest_teff[1] - closest_teff[0])
	yd = (logg - closest_logg[0])/(closest_logg[1] - closest_logg[0])

	# following wikipedia trilinear inteprlation - but cutting for only 2d
	c00 = h[0].data[indexs[0]]*(1-xd) + h[0].data[indexs[2]]*xd
	c01 = h[0].data[indexs[1]]*(1-xd) + h[0].data[indexs[3]]*xd

	c = c00*(1-yd) + c01*yd
	
	return 	create_spectrum_structure(h[0].data[0]/10, flux=c)
	


