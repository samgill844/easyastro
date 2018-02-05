from k2sc.ls import fasper
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def psearch(time, flux):
	freq,power,nout,jmax,prob = fasper(time, flux, 6, 0.5)
	period = 1./freq
	m = (period > 0.05) & (period < 35) 
	period, freq, power = period[m], freq[m], power[m]
	j = np.argmax(power)

	expy = np.exp(-power)
	effm = 2.*nout/6
	fap  = expy*effm

	mfap = fap > 0.01
	fap[mfap] = 1.0-(1.0-expy[mfap])**effm

	return period, power, fap, j

def periodigram(time,flux, font_size=12, plot='both'):
	matplotlib.rcParams.update({'font.size': font_size})

	periodp, powerp, fapp, jp = psearch(time,flux)
	print('Best period =',periodp[jp],", FAP = ",fapp[jp])
	if plot=='period':
		plt.plot(periodp,powerp, 'k')
		plt.xlabel('Period [d]')
		plt.ylabel('Power')
		plt.grid()
		plt.show()

	if plot=='fapp':
		plt.plot(periodp,fapp, 'k')
		plt.xlabel('Period [d]')
		plt.ylabel('FAP')
		plt.grid()
		plt.show()

	if plot=='both':
		fig,ax = plt.subplots(2,1, sharex=True)
		ax[0].plot(periodp,powerp, 'k')
		ax[0].grid()
		ax[0].set_ylabel('Power')

		ax[1].semilogy(periodp,fapp, 'k')
		ax[1].grid()
		ax[1].set_ylabel('FAP')
		plt.setp(ax[1], xlabel='Period [d]')
		fig.tight_layout()
		plt.show()

	if plot=='both_same':
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()

		ax1.plot(periodp,powerp, color='k')
		ax1.set_ylabel('Power')
		ax1.set_xlabel('Period [d]')

		ax2.semilogy(periodp,1 - fapp, color='r', ls='-', alpha = 0.2, )
		ax2.set_ylabel('1 - FAP')

		plt.show()


		

