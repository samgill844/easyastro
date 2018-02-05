#include<stdlib.h>
#include<stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <time.h>
#include <device_functions.h>

#define pi 3.14159265359
#define C  2.998e+8




/*
###############################################################################
#                           GPU WORK EQUATIONS                                #
###############################################################################
*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
###############################################################################
#                           GEOMETRY EQUATIONS                                #
###############################################################################
*/

float area(float d, float R1, float R2)
{
	/*
	Returns area of overlapping circles with radii x and R; separated by a distance d
	*/
	float arg1 = (d*d + R1*R1 - R2*R2)/(2.*d*R1);
	if (arg1 < -1)
	{
		arg1 = -1;
	}
	else if (arg1 > 1)
	{
		arg1 = 1;
	}

	float arg2 = (d*d + R2*R2 - R1*R1)/(2.*d*R2);
	if (arg2 < -1)
	{
		arg2 = -1;
	}
	else if (arg2 > 1)
	{
		arg2 = 1;
	}
	float arg3 = max((-d + R1 + R2)*(d + R1 - R2)*(d - R1 + R2)*(d + R1 + R2), 0.);

	if(R1 <= R2 - d) return pi*R1*R1;							//planet completely overlaps stellar circle
	else if(R1 >= R2 + d) return pi*R2*R2;						//stellar circle completely overlaps planet
	else return R1*R1*acos(arg1) + R2*R2*acos(arg2) - 0.5*sqrt(arg3);			//partial overlap
}





/*
###############################################################################
#                     KEPLERIAN EQUATIONS                                     #
###############################################################################
*/



float t_ecl_to_peri(float t_ecl, float ecc, float omega, float b, float radius_1, float p_sid)
{
	//////////////////////////////////////////////////////////////////////////////////
	// Calculate the time of periastron passage immediately prior to a give time of //
	// eclipse. Equation numbers from Hilditch, "An Introduction to Close Binary    //
	// Stars"																		//
	//////////////////////////////////////////////////////////////////////////////////
	/*
	float precision, intent (in) :: t_ecl ! Time of eclipse
	double precision, intent (in) :: ecc   ! Orbital eccentricity
	double precision, intent (in) :: omega ! Longitude of periastron, radians
	double precision, intent (in) :: incl  ! Orbital inclination, radians
	double precision, intent (in) :: p_sid ! Siderial period
	*/

	// Calculate inclination
	float incl = acos(b*radius_1);
	// Define variables used
	float  theta, theta_0, delta_t , ee, eta;
	float tol = 1e-5;


	float efac  = 1.0 - pow(ecc, 2);
	float sin2i = pow(sinf(incl), 2);

	// Value of theta for i=90 degrees
	theta_0 = (pi/2) - omega; // True anomaly at superior conjunction

	if (incl != pi/2)
	{
 		//par = (/ efac, sin2i, omega, ecc );
 		//d =  brent(theta_0-pi/2,theta_0,theta_0+pi/2, delta_func, npar, par, tol, theta, verbose1);
		memcpy(&theta, &theta_0, sizeof(float));
		
		// Parameters associated with minimization
		float sep = 100.0;
		float sep_trial;
		float step=0.1;
		int nit=0;
		float diff=100.0;
		while (diff > tol)
		{
			theta_0 = theta_0 + step;
			
			sep_trial = (1-powf(ecc,2)) * sqrt( 1 - powf(sinf(incl), 2) * powf(sinf(theta_0 + omega), 2)) / (1 + ecc*sin(theta_0));
			diff = abs(sep - sep_trial);
			if (sep_trial < sep)
			{
				// going in the right direction
				memcpy(&sep, &sep_trial, sizeof(float));
				memcpy(&theta, &theta_0, sizeof(float));
				//printf("\n-B nit: %d  theta: %f  sep: %f   step: %f", nit, theta, sep, step);
			}
			else
			{
				// going in the wrong direction, reverse
				memcpy(&sep, &sep_trial, sizeof(float));
				memcpy(&theta, &theta_0, sizeof(float));
				step = -step/2;
				//printf("\n-G nit: %d  theta: %f  sep: %f   step: %f", nit, theta, sep, step);
			}

			//printf("\n diff: %f", diff);
			//fflush(stdout);
			nit ++;
		
		}
	}
	else
	{
		memcpy(&theta, &theta_0, sizeof(float));
	}


	if (theta == pi)
	{
 		ee = pi;
	}
	else
	{
 		ee = 2.0 * atanf(sqrt((1.0-ecc)/(1.0+ecc)) * tanf(theta/2.0));
	}
	
	eta = ee - ecc*sinf(ee);
	delta_t = eta*p_sid/(pi*2);
	//printf("\nReturn: %f", t_ecl  - delta_t);
	return  t_ecl  - delta_t;
	
}

float eanom(float m, float e)
{
    /*
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
    */
    int itmax = 9999;
    float etol=1.0e-5;


    int it = 0;
    float e1 = fmod(m,2*pi) + e*sin(m) + e*e*sin(2.0*m)/2.0;
    float e0;
    float test = 1.0;
    while (test > etol)
    {
        it ++;
        e0 = e1;
        e1 = e0 + (m-(e0 - e*sin(e0)))/(1.0 - e*cos(e0));
        test = abs(e1 - e0);

        if (it > itmax)
        {
            printf("FAIL");
            return -1;
        }
    }

    if (e1 < 0)
    {
        e1 = e1 + 2*pi;
    }
    return e1;
}


float trueanom(float m, float e)
{
    /*
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
    */

    float ee = eanom(m,e);
    return 2.0*atan(sqrt((1.0 + e)/(1.0 - e))*tan(ee/2.0));
}

float light_travel_time( float a_p, float incl, float e, float theta, float omega)
{
    /*
    Caluclate light travel time using the projected semi-major axis. 
	Uses Eqn. 2.57 from Hilditch
 
	Input:
		a_p : projected semi-major axis
		incl : inclination (rad)
		e : eccentricity
		theta : true anomaly
		omega : argument of periastron

	Output:
		z / c : 
    */

	
	return a_p*sinf(incl)*(1-pow(e,2))*sinf(theta + omega) / (C * (1 + e*cosf(theta)));
}


int closeset_star( float theta, float omega, float incl)
{
	/*
	Calculate the closest star using the z coord

	*/
	float theta_2 = theta + pi; 

	float z1 = sinf(theta + omega)*sinf(incl);
	float z2 = sinf(theta_2 + omega)*sinf(incl);

	//printf("\n z1 : %f  z2 : %f", z1,z2);      
	if (z1<z2)
	{
		return 1; // star 2 is infront of star 1 (primary eclipse)
	}
	else
	{
		return 0; // star 1 is infront of star 2 (secondary eclipse)
	}
} 





/*
###############################################################################
#                     RADIAL VELOCITY EQUATIONS                               #
###############################################################################
*/



void _radvel(float t, float t0, float p, float v0, float dv0, float k1, float k2, float f_c, float f_s, float *RV1, float *RV2, int index)
{
    /*
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
    */
    float e = sqrt(f_c*f_c + f_s*f_s);
	float w = atanf(f_s/f_c);
	
    float m = 2*pi*fmod((t - t_ecl_to_peri(t0, e, w,  0, 0.2, p))/p,1.0);

    float omrad = atan(f_s / f_c);
    if (e == 0.0)
    {
        RV1[index] =  v0 +k1*cos(m+omrad) + dv0*(t-t0);
        RV2[index] =  v0 +k2*cos(m+omrad + pi) + dv0*(t-t0);
    }
    else
    {
        RV1[index] =  v0 + k1*( e*cos(omrad) + cos(trueanom(m,e) + omrad)) + dv0*(t-t0);
        RV2[index] =  v0 + k2*( e*cos(omrad) + cos(trueanom(m,e)+pi + omrad)) + dv0*(t-t0);
    }
}

extern "C" {
void radvel(float t[], float t0, float p, float v0, float dv0, float k1, float k2, float f_c, float f_s, float *RV1, float *RV2, int n)
{
    for (int i=0; i<n; i++)
    {
        _radvel(t[i], t0, p, v0, dv0, k1, k2, f_c, f_s, RV1, RV2, i);
    } 
}
}



/*
###############################################################################
#                         LIMB DARKENING LAWS                                 #
###############################################################################
*/

float get_I_from_limb_darkening(int ld_law, float ldc[], float mu_i)
{
	/*
	Calculte limb-darkening for a variety of laws e.t.c.

	[0] linear (Schwarzschild (1906, Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse, p. 43)
	[1] Quadratic Kopal (1950, Harvard Col. Obs. Circ., 454, 1)
	[2] Square-root (Díaz-Cordovés & Giménez, 1992, A&A, 259, 227) 
	[3] Logarithmic (Klinglesmith & Sobieski, 1970, AJ, 75, 175)
	[4] Exponential LD law (Claret & Hauschildt, 2003, A&A, 412, 241)
	[5] Sing three-parameter law (Sing et al., 2009, A&A, 505, 891)
	[6] Claret four-parameter law (Claret, 2000, A&A, 363, 1081)
	*/
	switch (ld_law)
	{
		case 0 : return 1 - ldc[0]*(1 - mu_i); 
		case 1 : return 1 - ldc[0]*(1 - mu_i) - ldc[1]*pow((1 - mu_i),2); 
		case 2 : return 1 -  ldc[0]*(1 - mu_i) - ldc[1]*(1 - pow(mu_i,0.5)); 
		case 3 : return 1 -  ldc[0]*(1 - mu_i) - ldc[1]*mu_i*log(mu_i); 
		case 4 : return 1 -  ldc[0]*(1 - mu_i) - ldc[1]/(1-exp(mu_i));  
		case 5 : return 1 -  ldc[0]*(1 - mu_i) - ldc[1]*(1 - pow(mu_i,1.5)) - ldc[2]*(1 - pow(mu_i,2));
		case 6 : return 1 - ldc[0]*(1 - pow(mu_i,0.5)) -  ldc[1]*(1 - mu_i) - ldc[2]*(1 - pow(mu_i,1.5))  - ldc[3]*(1 - pow(mu_i,2));
		default : return 1 - ldc[0]*(1 - mu_i);
	}
}



/*
###############################################################################
#                        LIGHT CURVE EQUATIONS                                #
###############################################################################
*/

__global__ void  d_lc(float *t, float t0, float p, float radius_1, float k, float incl, float *I, size_t ss)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < ss)
	{
		// Calculate the sky-projected seperation
		float z = 3*sqrt( 1 - cosf(2*pi*(t[idx] - t0)/p)*cosf(2*pi*(t[idx] - t0)/p)*sinf(incl)*sinf(incl));

		// define the number of radial segments
		int n = 1000;
		float dr = radius_1/n;

		// Define the flux of the star regardless of occulation
		float I_star_ = 0.0; // this is going to be the total stellar flux
		float F_occ_ = 0.0;
		float test,F_occ;


		// cycle the number of radial components
		for (int i = 0; i< n; i++)
		{
		    // calculate r_i
		    float r_i = (0.5 + i)/n ; 

		    // calculate mu_i
		    float mu_i = sqrt(1 - r_i*r_i);

		    // Calculate I(r_i, m_i)
		    float I_i = 1 - 0.6*(1 - mu_i);


		    // Calculate the flux over area dr
		    float F_i = I_i*2*pi*r_i*dr; // the flux across the ring
		    I_star_ += F_i; //append it to get the total flux 

		    // Calculate the occulted flux
		    F_occ = 0.0;
		    if (r_i <= (z-k) || r_i >= (k+z))
		    {
		        F_occ = 0.0;
		    }
		    else if (r_i > abs(z-k) && r_i < (z+k))
		    {
		        test = (r_i*r_i + z*z - k*k)/(2*z*r_i);
		        if (test >= 1.0)
		        {
		            test = 1.0;
		        }
		        if (test <= -1.0)
		        {
		            test = -1.0;
		        }
		        F_occ = acosf(test)/pi;
		    }
		    else if (r_i <= (k-z))
		    {
		        F_occ = 1.0;
		    }
		    F_occ_ += F_occ*2*pi*r_i*dr; // 
		}

        I[idx] =  1-F_occ_/I_star_;
    }
	
}

float lc(float t, float t0, float p, float radius_1, float k, float b, float f_c, float f_s, int ld_law_1,  float * ldc_1, float S)
{
	// Calculate e and w
	float e = sqrt(f_c*f_c + f_s*f_s);
	float w;
	if (f_c == 0.0) w = 0.0;
	else  w = atanf(f_s / f_c);

	// Calculate inclination
	float incl = acos(b*radius_1);
	//printf("incl: %f", incl);

	// Calculate the mean anomaly
	float m = 2*pi*fmod((t - t_ecl_to_peri(t0, e, w,  b, radius_1, p))/p,1.0);

	// Calculate the true anomaly
	float mu = trueanom(m, e);

	// Calculate the projected seperation in units of a.
	float a =  4.20944 *pow(p, 2./3.) * pow(2, 1./3.) ;
	float z;
	int type_of_transit;
	if (f_c==0 && f_s==0)
	{
		// If we are here it is because the orbit is circular
    	z = a*sqrt( 1 - cosf(2*pi*(t - t0)/p)*cosf(2*pi*(t - t0)/p)*sinf(incl)*sinf(incl)); 
	}
	else
	{
		// If we are here it is because the orbit is eccentric
		if (f_c==0)
		{
			w = 0;
		}

        ///return (1-e**2)*np.sqrt(1-np.sin(i)**2*np.sin(mu + w)**2 ) / (1 + e*np.cos(mu)) 
    	z = a*(1-powf(e,2)) * sqrt( 1 - powf(sinf(incl), 2) * powf(sinf(mu + w), 2)) / (1 + e*sin(mu)) ; 
	}

	//return z;
    // define the number of radial segments
    int n = 1000;
    float dr = 1.0/n;

    // Define the flux of the star regardless of occulation
    float I_star_ = 0.0; // this is going to be the total stellar flux
    float F_occ_ = 0.0;
    float test, F_occ;
    // cycle the number of radial components
    for (int i = 0; i< n; i++)
    {
        // calculate r_i
        float r_i = (0.5 + i)/n ; 

        // calculate mu_i
        float mu_i = sqrt(1 - r_i*r_i);

        // Calculate I(r_i, mu_i) with linear limb-darkening law
        //float I_i = 1 - 0.6*(1 - mu_i);
		//const char *ld_law = "lin";
		float I_i = get_I_from_limb_darkening(ld_law_1, ldc_1 , mu_i);

        // Calculate the flux over area dr
        float F_i = I_i*2*pi*r_i*dr; // the flux across the ring

		//append it to get the total flux
        I_star_ += F_i; 

        // Calculate the occulted flux
        if (r_i <= (z-k) || r_i >= (k+z))
        {
            F_occ = 0.0;
        }
        else if (r_i > abs(z-k) && r_i < (z+k))
        {
            test = (r_i*r_i + z*z - k*k)/(2*z*r_i);
            if (test >= 1.0)
            {
                test = 1.0;
            }
            if (test <= -1.0)
            {
                test = -1.0;
            }
            F_occ = acosf(test)/pi;
        }
        else if (r_i <= (k-z))
        {
            F_occ = 1.0;
        }
		else
		{
			printf("What");
		}

		// Now check if star 2 is occulting star 1
		//printf("\nmu : %f  w : %f incl : %f", mu, w, incl) ;
		type_of_transit = closeset_star( mu, w, incl);
		switch (type_of_transit)
		{
			case 0: // primary eclipse
        		F_occ_ += F_i*F_occ; // *2*pi*r_i*dr; // 
			
			case 1: // secondary eclipse
				F_occ_ += S * F_i * area(z, radius_1, k*radius_1);				
		}



    }
    //printf("\nF0cc: %f\n I_star_: %f\n", F_occ_, I_star_);
    //printf("r_i: %f\n dr: %f\n", dr, dr);
    return 1 - F_occ_/I_star_;
}



extern "C" {
void cuda_lc(float *t, float t0, float p, float radius_1, float k, float b, float f_c, float f_s, int ld_law_1, float *ldc_1, float S,  float *I, int n, char *CPUorGPU)
{
    int devices = 0; 
    cudaError_t err = cudaGetDeviceCount(&devices); 

    const char *GPU = "GPU";
    const char *CPU = "CPU";

    // Check number of PCI's to see if we should use CPU or GPU
    if (devices > 0 && err == cudaSuccess && strcmp(CPUorGPU, GPU) == 0 ) 
    { 
        // Run CPU+GPU code
		//printf("Using GPU");
		// alloc pointers
		float  *d_t, *d_I;

		// Cuda Malloc each parameter	
		cudaMalloc( &d_t, n*sizeof(float));					
		cudaMalloc( &d_I, n*sizeof(float));	

		cudaMemcpy( d_t, t, n*sizeof(float), cudaMemcpyHostToDevice);																									
		cudaMemcpy( d_I, I, n*sizeof(float), cudaMemcpyHostToDevice);

		d_lc <<< ceil(n / 256.0), 256 >>> (d_t, t0, p, radius_1, k, b, d_I, n);

		cudaMemcpy( I, d_I, n*sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_t);
		cudaFree(d_I);
    } 
    else if (strcmp(CPUorGPU, CPU) == 0)
    { 
		//printf("Using CPU");
        for (int i=0; i<n; i++)
        {
            I[i] = lc(t[i], t0, p, radius_1, k, b, f_c, f_s, ld_law_1,  ldc_1, S);
        }    
    } 
    else
    {
    perror("I couldn't decide whether to use the the GPU or CPU");
    }

 
}
} 



