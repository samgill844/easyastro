from numba import jit
import numpy as np
import matplotlib.pyplot as plt

@jit(nopython=True)
def eanom(m,e):
    '''
    !  Calculate the eccentric anomaly of a Keplerian orbit with
    ! eccentricity e, from the mean anomaly, m.
    !
    !  Solves Kepler''s equation using Newton-Raphson iteration using formula for
    ! initial estimate from Heintz DW, 'Double stars' (Reidel, 1978).
    ! 
    !  Input:
    !   m - Mean anomaly in radians.
    !   e - Eccentrcity, 0 <= E < 1.
    ! 
    !  Output:
    !    Eccentric anomaly in the range 0 < eanom < 2*pi.
    !    If e is out-of-range return bad_dble
    !
    '''
    itmax = 9999
    etol=1.0e-9



    it = 0
    e1 = np.fmod(m,2*np.pi) + e*np.sin(m) + e*e*np.sin(2.0*m)/2.0
    test = 1.0
    while (test > etol):
        it = it + 1
        e0 = e1
        e1 = e0 + (m-(e0 - e*np.sin(e0)))/(1.0 - e*np.cos(e0))
        test = abs(e1 - e0)

        if (it > itmax):
            print('FAIL', e0,e1,m,e)
            return -1

    if (e1 < 0):
        e1 = e1 + 2*np.pi

    return e1

@jit(nopython=True)
def trueanom(m, e):
    '''
    !  Calculate the true anomaly of a Keplerian orbit with eccentricity e,
    ! from the mean anomaly, m.
    !
    ! Uses: eanom
    !
    !  Input:
    !   m - Mean anomaly in radians.
    !   e - Eccentrcity, 0 <= e < 1.
    !
    ! Output:
    !   True anomaly in the range 0 to 2*PI.
    !   If e is out-of-range return bad_dble
    !  
    '''

    ee = eanom(m,e)
    return 2.0*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(ee/2.0))



@jit(nopython=True)
def _radvel(t, t0, p, v0, dv0, k1,k2, e, omrad):
    '''
    ! Calculate radial velocity for a Keplerian orbit
    !
    ! Uses: TRUEANOM
    !
    ! Input:
    !  T      - Time of observation
    !  T0     - Time of periastron
    !  P      - Orbital period
    !  V0     - Systemic velocity
    !  K      - Velocity semi-amplitude in the same units as V0
    !  E      - Eccentricity of the orbit
    !  OMRAD  - Longitude of periastron in radians.
    !
    !  Output:
    !   Radial velocity in the same units as V0.
    !
    '''
    m = 2*np.pi*np.fmod((t-t0)/p,1.0)
    if (e == 0.0):
        return v0 +k1*np.cos(m+omrad) + dv0*(t-t0), 0 +k2*np.cos(m+omrad + np.pi) + dv0*(t-t0)

    else:
        return  v0 + k1*( e*np.cos(omrad) + np.cos(trueanom(m,e) + omrad)) + dv0*(t-t0) , v0 + k2*( e*np.cos(omrad) + np.cos(trueanom(m,e)+np.pi + omrad)) + dv0*(t-t0)



@jit(nopython=True)
def calculate_radial_velocity(time, period=1,T0=0,K1=5, K2 = 1,e=0, w=0.0, V0=0, dV0 = 0) :
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
    rv1 = np.zeros(time.shape[0])
    rv2 = np.zeros(time.shape[0])
    for i in range(time.shape[0]):
        rv1[i],rv2[i] = _radvel(t=time[i], t0=T0, p=period, v0=V0, dv0=dV0, k1=K1, k2=K2, e=e, omrad = w)
    return rv1,rv2





'''
#@jit(nopython=True)
def calculate_radial_velocity(time,period=1,T0=0,K1=5, K2 = 1,e=0,w=0.5*np.pi,V0=0, dV0 = 0, rossiter = False,  Vsini= 5.0, beta = np.pi, i = 89.9*np.pi/180, u_1 = 0.6, r1_a=0.1, r2_a=0.05):
    
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
    
    
    

    RV1 = np.zeros(len(time))
    RV2 = np.zeros(len(time))

    for i in range(len(time)):
        ##################################
        # First define the Mean anomaly
        ##################################
        M =  2*np.pi* np.fmod( 1.00 + np.fmod((time[i]-time[0])/ period , 1.00), 1.00  ) 
#            np.abs(time[i]-time[0]))/period

        ####################################################################################
        # Now solve Keplers equation via Newton Raphson iteration for the eccentric anomaly
        ####################################################################################
        E = newton_raphson(M,e)

        ################################################
        # Now calculate the true anomaly, v1
        # RV = 2 * atan( sqrt( (1 + e)/(1-e)) tan(E/2))
        ################################################
        v1 = 2*np.arctan2(np.sqrt((1+e)/(1-e))*np.tan(E/2))
        v2 = v1 + np.pi
        
        ####################################
        # Now calculate the radial velocity
        # RV = K* ( cos(w + v1) + e*cos(w)) 
        ####################################
        RV1[i] = K1*(np.cos(v1 + w) + e*np.cos(w)) + dV0*(time[i]-T0) +V0
        RV2[i] = K2*(np.cos(v2 + w) + e*np.cos(w)) + dV0*(time[i]-T0) + V0

        ###############################################
        # If Rossiter == True, get the RM measurements
        ###############################################
        if rossiter == True:
            #################################################
            # Following notation of Giménez_2006_ApJ_650_408
            #################################################

            ######################################
            # Calculate phase of the observation
            ######################################
            phase = ((time[i] - T0) / period ) - np.floor((time[i] - T0) / period )
            phase = 2*np.pi*phase 


            #########################################################
            # Now using eqn. 14 to calculate the serperation, delta
            #########################################################
            delta = np.sqrt(      (1-e**2)**2 / (1 - e*np.sin(phase - w))**2     * (1 - np.cos(phase)**2 * np.sin(i)**2)   )
            

            ##############################
            # Now check if its transiting
            ##############################
            r_sum = r1_a + r2_a
            theta_start = np.arccos( np.sqrt( (1 - (r1_a - r2_a)**2) / np.sin(i)**2 ) )
            if delta > r_sum:
                continue

            elif delta < r_sum:


                ###########################
                # Now caluclate V_ (eqn 4)
                # beta : projection on the plane of the sky with that of the pole of the orbit
                ###########################
                V_ = Vsini * (np.sin(beta)*np.cos(i)*np.cos(phase) - np.cos(beta)*np.sin(phase))



                ################################################
                # Now use eqn. 16 for linear limb-darkening law
                ################################################
                dV_rm = (V_ / delta) * (1 - u_1) / (1 - u_1 + 2*u_1/3)
            
                #print('Phase: {}        dV_rm: {}'.format(phase, dV_rm))
                RV1[i] = RV1[i] - dV_rm

                print('V_: {}  delta: {}  dV_rm: {}  theta_start: {}'.format(V_, delta,dV_rm,theta_start))

    return RV1, RV2

'''


