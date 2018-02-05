from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import scipy



###############################################################################
#                     KEPLERIAN EQUATIONS                                     #
###############################################################################

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






###############################################################################
#                           RADIAL VELOCITY                                   #
###############################################################################

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





###############################################################################
#                         LIGHTCURVE MODEL                                    #
###############################################################################
@jit(nopython=True) 
def integrate(radius_1, k, delta,shortest_delta,delta_x, n_r, n_phi, dr, dphi):
    '''
    Integrate using spider-web geometry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    radius_1 : float
        The ratio of radii 1 to the semi-major axis.
    k : float
        The ratio of the radii.
    delta : float
        The projected seperation in units of semi-major axis.
    n_r : int
        The number of radial bisections.
    n_phi : int
        The number of angular bisections.
    '''
    #Step 1
    #   - Set up the gemoetry
    theta = np.arccos(shortest_delta/delta)
    I=0
    for i in range(n_r):
        for j in range(n_phi):
            ########################
            # Part 2
            ########################
            # bisect middle of radiall compenent to get r_i
            r_i = radius_1*((2*i+1))/(2*n_r)
    
            # bisect iddle of phi to get phi_i
            phi_i = 2*np.pi*((2*j+1))/(2*n_phi)
        

            ########################
            # Part 4
            ########################   
            # We now need to be careful with angles and test to see which
            # quadrant we are in
            #print(mu, shortest_mu, phi_i, r_i)
            case=-1
            if (phi_i < np.pi/2): # quad 1
                r_i_x = -r_i*np.sin(phi_i)
                r_i_y = r_i*np.cos(phi_i)
                case=1
            elif (phi_i > np.pi/2) and (phi_i < np.pi)  : # quad 2
                r_i_y = -r_i*np.sin(phi_i-np.pi/2)
                r_i_x = -r_i*np.cos(phi_i-np.pi/2)
                case=2
            elif (phi_i > np.pi) and (phi_i < np.pi*1.5)  : # quad 3
                r_i_x = r_i*np.sin(phi_i-np.pi)
                r_i_y = -r_i*np.cos(phi_i-np.pi)
                case=3
            elif (phi_i > np.pi*1.5) and (phi_i < 2*np.pi)  : # quad 4
                r_i_y = r_i*np.sin(phi_i-np.pi*1.5)
                r_i_x = r_i*np.cos(phi_i-np.pi*1.5)
                case=4
            else:
                raise ValueError('Wheres r_i_x and r_i_y')
                
            ########################
            # Part 5
            ########################               
            # now check to see if coordinate of r_i_x and r_1_y is within
            # the circle of r_2
            #print(((r_i_x - delta_x)**2 , (r_i_y + shortest_delta)**2 ) , k*radius_1)
            #print(r_i_x, delta_x)
            if (((r_i_x - delta_x)**2 + (r_i_y + shortest_delta)**2 ) < k*radius_1):
                # point is occulted/ and the flux is from star 2
                f_star_2 = 0.0
                I += f_star_2
            else:
                mu = np.sqrt(1-r_i**2)
                # dc = d_phi*dr
                #    = 2*pi/phi * dr
                da = dr * 2*np.pi/dphi
                I += 1-0.6*(1-mu)*da
                
    return I
        
        
        
    
def calculate_delta(mu, e, i, w):
    return (1-e**2)*np.sqrt(1-np.sin(i)**2*np.sin(mu + w)**2 ) / (1 + e*np.cos(mu)) 


    
#@jit(nopython=True)   
def _calculate_lc(time, period, t0, k, radius_1, b, e, w, SBR):
    # Step 1 - calculate the mean anomaly
    m = 2*np.pi*np.fmod((time-t0)/period,1.0)

    # Step 2 - calculate the true anomaly
    mu = trueanom(m, e)
    #print(mu*180/np.pi)
    
    # Step 3 - calculate inclination
    incl = (180*np.arccos(b*radius_1)/np.pi)*np.pi/180
    #print('Incl: {}'.format(incl))
    
    # Step 4 - calculate the projected seperation (Hilditch Eqn. 5.63)
    w = (w - 90)*np.pi/180 # convert to radians
    #w = w*np.pi/180
    delta = calculate_delta(e, incl, mu, w)
    
    # Step 5 - Check if transiting 
    if (delta < (radius_1 + radius_1*k)):
        ########################
        # Part 1
        ########################
        # set up geometry
        n_r = 100
        n_phi = 100
        dr = radius_1/n_r
        dphi = 2*np.pi/n_phi
        #print(radius_1)
        # Now start integrating from systems "north" and integrate round starting
        # from the outer ring.         
        ########################
        # Part 3
        ########################
        # solve for shortest m uand thus delta
        shortest_mu = scipy.optimize.minimize(calculate_delta, mu, args=(e,incl,w), method = 'L-BFGS-B').x[0]
        shortest_delta = calculate_delta(shortest_mu, e,incl,w)
        if delta>shortest_delta:
            delta_x = np.abs(np.sqrt(delta**2 - shortest_delta**2))
        else:
            delta_x = delta
        #print(delta, delta_x, shortest_delta)
        '''
        # now find theta
        # cos(theta) = delta(mu_0) /delta
        theta = np.arccos(shortest_delta/delta)
        for i in range(n_r):
            for j in range(n_phi):
                ########################
                # Part 2
                ########################
                # bisect middle of radiall compenent to get r_i
                r_i = radius_1*((2*i+1))/(2*n_r)
        
                # bisect iddle of phi to get phi_i
                phi_i = 2*np.pi*((2*j+1))/(2*n_phi)
            

                ########################
                # Part 4
                ########################   
                # We now need to be careful with angles and test to see which
                # quadrant we are in
                #print(mu, shortest_mu, phi_i, r_i)
                if (phi_i < np.pi/2): # quad 1
                    r_i_x = r_i*np.sin(phi_i)
                    r_i_y = r_i*np.cos(phi_i)
                elif (phi_i > np.pi/2) and (phi_i < np.pi)  : # quad 2
                    r_i_y = r_i*np.sin(phi_i-np.pi/2)
                    r_i_x = r_i*np.cos(phi_i-np.pi/2)
                elif (phi_i > np.pi) and (phi_i < np.pi*1.5)  : # quad 3
                    r_i_x = r_i*np.sin(phi_i-np.pi)
                    r_i_y = r_i*np.cos(phi_i-np.pi)
                elif (phi_i > np.pi*1.5) and (phi_i < 2*np.pi)  : # quad 4
                    r_i_y = r_i*np.sin(phi_i-np.pi*1.5)
                    r_i_x = r_i*np.cos(phi_i-np.pi*1.5)
                else:
                    raise ValueError('Wheres r_i_x and r_i_y')
                    
                ########################
                # Part 5
                ########################               
                # now check to see if coordinate of r_i_x and r_1_y is within
                # the circle of r_2
                if (((r_i_x - delta_x)**2 + (r_i_y - shortest_delta)**2 ) < k*radius_1):
                    # point is occulted/ and the flux is from star 2
                    f_star_2 = 0.0
                    I += f_star_2
                else:
                    mu = np.sqrt(1-r_i**2)
                    
                    # dc = d_phi*dr
                    #    = 2*pi/phi * dr
                    da = dr * 2*np.pi/dphi
                    I += 1-0.6*(1-mu)*da
                    print(I)

                    
        '''       
        #print(integrate(radius_1, k, delta,shortest_delta,delta_x, n_r, n_phi, dr, dphi))
        return integrate(radius_1, k, delta,shortest_delta,delta_x, n_r, n_phi, dr, dphi)


            
    else:
        # Else fif not transiting, return a flux of 1
        return 1
        
    
    
    


#@jit(nopython=True)
def calculate_LC(time, period=1,T0=0, k = 0.2, radius_1 = 0.2, b = 0.05, f_c=0.1, f_s=0.0, SBR=0.1):
    '''
    Calulates the lightcurve model for a binary system for a given set of 
    parameters.
    
    Parameters
    -----------
    time : nump array
        An array describing the time stamps at which to calculate the light-
        curve model.
    period : float
        The period of the binary system.
    T0 : float
        The epoch at which the primary star transits. 
    k : float
        The fractional radii (R_2 / R_1).
    radius_1 : float
        The fractional radius of star 1 (R_1/a; where a is the semi-major 
        axis).
    b : float
        The impact parameter
    '''
    if (f_c==0) and (f_s==0):
        e=0
        w=0
    else:
        e = f_c**2 + f_s**2
        w = np.arctan2(f_s, f_c)
    

    lightcurve = np.zeros(time.shape[0])
    mu = np.zeros(time.shape[0])
    for i in range(time.shape[0]):
        print(100*i/time.shape[0])
        lightcurve[i] = _calculate_lc(time[i], period,T0, k , radius_1, b, e=e, w=w, SBR = SBR)
                  
    return lightcurve
    

t = np.linspace(8,45,1000)
plt.plot(t, calculate_LC(t, period=20, f_c=0.1, f_s=0., b=0.)/9991.951504763343)
plt.grid()
plt.show()




