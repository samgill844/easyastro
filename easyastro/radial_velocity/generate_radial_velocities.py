from numba import jit
import numpy as np

@jit(nopython=True)
def kepler(E,M,e):
	return E - e*np.sin(E) - M

@jit(nopython=True)
def dkepler(E,e):
	return 1 - e*np.cos(E)

@jit(nopython=True)
def newton_raphson(M,e):
    ####################################
    # Keplers equation
    # f(E) = E - e*np.sin(E) - M = 0
    # and the differential form 
    # df(E) = 1 - e*np.cos(E)
    #
    # We use the Newton raphson method
    #####################################
	E = 1
	for j in range(10):
		E = E - kepler(E,M,e)/dkepler(E,e)
	return E
    


@jit(nopython=True)
def calculate_radial_velocity(time,period=1,T0=0,K1=5, K2 = 1,e=0,w=0.5*np.pi,V0=0, dV0 = 0):
    '''
    Calculates radial velocities for a binary system (both components)
    for a given set of parameters and time series. 
    
    Parameters
    ----------
    time : nump array
        An array describing the time stamps at which to calculate the radial 
        velocities.
    period : float
        The period of the binary system.
    T0 : float
        The epoch at which the primary star transits. 
    K1 : float
        The semi-amplitude of the RV measurements for star 1 (km/s).
    K2 : float
        The semi-amplitude of the RV measurements for star 2 (km/s).
    e : float
        The eccentricity of the system.
    w : The argument of the periastron (in radians NOT degrees). Unused if 
        eccentricity is set to 0.
    V0 : float
        The systematic velocity of the binary system. 
    dV0 : float
        The change in systematic velocity of the binary system with time.
        
    Returns
    -------
    rv1 : numpy array
        The radial velocity measurements for star 1 corrosponding to input
        time.
    rv2 : numpy array
        The radial velocity measurements for star 2 corrosponding to input
        time.
    
    
    '''

    RV1 = np.zeros(len(time))
    RV2 = np.zeros(len(time))
    for i in range(len(time)):
        ##################################
        # First define the Mean anomaly
        ##################################
        M = 2*np.pi*(time[i]-time[0])/period

        ####################################################################################
        # Now solve Keplers equation via Newton Raphson iteration for the eccentric anomaly
        ####################################################################################
        E = newton_raphson(M,e)
        ################################################
        # Now calculate the true anomaly, v1
        # RV = 2 * atan( sqrt( (1 + e)/(1-e)) tan(E/2))
        ################################################
        v1 = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        v2 = v1 + np.pi
        
        ####################################
        # Now calculate the radial velocity
        # RV = K* ( cos(w + v1) + e*cos(w)) 
        ####################################
        RV1[i] = K1*(np.cos(v1 + w) + e*np.cos(w)) + dV0*(time[i]-T0) +V0
        RV2[i] = K2*(np.cos(v2 + w) + e*np.cos(w)) + dV0*(time[i]-T0) + V0

    return RV1, RV2