#####################################################################################
#                              easyastro V-0.1                                      #
#####################################################################################

Python source code dedicated to the Easy analysis of asttronomical data.





Update log
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

V 0.1
~~~~~
The first implementation of easy astro.

The current structure is:


easyastro
|------------ lightcurves: A package dedicated to transit lightcurves.
|                  |
|                  |---- plan_when_transits_will_occur : a package to plan when to observe binary transits. 
|
|
|------------ radial_velocity : A package for the analysis of radial velocity measurements.
|                  |
|                  |---- calculate_radial_velocity : calculate the radial velocity given parameters of an RV curve
|                  |                                 and time stamps (time, e, w, K1, V0, dv0).
|                  |
|                  |---- fit_SB1 : fit the radial velocity curve for single lined eclipsing binaries. Uses the emcee
|------------------|               package to explore parameters space in a Monte Carlo fashion. 
