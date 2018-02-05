from astropy.time import Time
from astropy import coordinates as coord, units as u
from astropy.table import Table, Column
from astroplan import Observer
from astroplan import FixedTarget
from astropy.coordinates import  AltAz
from astroplan.plots import dark_style_sheet,plot_airmass
from astroplan import AltitudeConstraint, AirmassConstraint,AtNightConstraint
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
from astroplan.plots import plot_finder_image
from astroplan import is_observable, is_always_observable, months_observable
from matplotlib import dates
from pylab import *

def xrange(x):
    return iter(range(x))

def plan_when_transits_will_occur(filename = 'targets.txt',observatory='Southern African Large Telescope',start='2017-06-22',end='2017-06-28',airmass_limit=2.5,moon_distance=10,do_secondary=True, method = 'by_night'):
    '''
    Plan when targets will be visibile and transiting from a site. 
    
    Inputs
    ------
    filename : str
        A plain text file with the following columns:
            target : The name of the target (e.g. J0555-57).
            RA     : The right ascension of the target (e.g. 05h55m32.62s).
            DEC    : The declination of the target (e.g. -57d17m26.1s).
            epoch* : The epoch of the transit. Youc can either use:
                         epoch_HJD-2400000 : HJD - 24500000
                         epoch_BJD-2455000 : MJD
            Period : The period of the system (days).
            Secondary : can be True or False depending on whether you want
                        to see when the secondary transits will be.
    observatory : str
        The observatory you are observing from. See later for list of available
        observatories (accepted by astropy).    
    start : str
        The first night of observation (e.g. 2017-08-31).
    end : str
        The last night of observation (e.g. 2017-09-10).
    airmass_limit : float
        The maximum airmass you want to observe through. 
    moon_distance : float
        The closest the target can be t the moon in arcmins.
    do_secondary = True:
        Look for secondary eclipses assuming circularised orbits. 
        
    Available observator names are:
         'ALMA',
         'Anglo-Australian Observatory',
         'Apache Point',
         'Apache Point Observatory',
         'Atacama Large Millimeter Array',
         'BAO',
         'Beijing XingLong Observatory',
         'Black Moshannon Observatory',
         'CHARA',
         'Canada-France-Hawaii Telescope',
         'Catalina Observatory',
         'Cerro Pachon',
         'Cerro Paranal',
         'Cerro Tololo',
         'Cerro Tololo Interamerican Observatory',
         'DCT',
         'Discovery Channel Telescope',
         'Dominion Astrophysical Observatory',
         'Gemini South',
         'Hale Telescope',
         'Haleakala Observatories',
         'Happy Jack',
         'Jansky Very Large Array',
         'Keck Observatory',
         'Kitt Peak',
         'Kitt Peak National Observatory',
         'La Silla Observatory',
         'Large Binocular Telescope',
         'Las Campanas Observatory',
         'Lick Observatory',
         'Lowell Observatory',
         'Manastash Ridge Observatory',
         'McDonald Observatory',
         'Medicina',
         'Medicina Dish',
         'Michigan-Dartmouth-MIT Observatory',
         'Mount Graham International Observatory',
         'Mt Graham',
         'Mt. Ekar 182 cm. Telescope',
         'Mt. Stromlo Observatory',
         'Multiple Mirror Telescope',
         'NOV',
         'National Observatory of Venezuela',
         'Noto',
         'Observatorio Astronomico Nacional, San Pedro Martir',
         'Observatorio Astronomico Nacional, Tonantzintla',
         'Palomar',
         'Paranal Observatory',
         'Roque de los Muchachos',
         'SAAO',
         'SALT',
         'SRT',
         'Siding Spring Observatory',
         'Southern African Large Telescope',
         'Subaru',
         'Subaru Telescope',
         'Sutherland',
         'Vainu Bappu Observatory',
         'Very Large Array',
         'W. M. Keck Observatory',
         'Whipple',
         'Whipple Observatory',
         'aao',
         'alma',
         'apo',
         'bmo',
         'cfht',
         'ctio',
         'dao',
         'dct',
         'ekar',
         'example_site',
         'flwo',
         'gemini_north',
         'gemini_south',
         'gemn',
         'gems',
         'greenwich',
         'haleakala',
         'irtf',
         'keck',
         'kpno',
         'lapalma',
         'lasilla',
         'lbt',
         'lco',
         'lick',
         'lowell',
         'mcdonald',
         'mdm',
         'medicina',
         'mmt',
         'mro',
         'mso',
         'mtbigelow',
         'mwo',
         'noto',
         'ohp',
         'paranal',
         'salt',
         'sirene',
         'spm',
         'srt',
         'sso',
         'tona',
         'vbo',
         'vla'.
    '''
	###################
	# Try reading table
	###################
    try:
        target_table = Table.read(filename,format='ascii')
    except:
        raise ValueError('I cant open the target file (make sure its ascii with the following first line:\ntarget		RA		DEC		epoch_HJD-2400000	Period		Secondary')

	##############################
	# try reading observation site
	##############################
    try:
        observation_site = coord.EarthLocation.of_site(observatory) 
        observation_handle = Observer(location=observation_site)
        observation_handle1 = Observer.at_site(observatory)
    except:
        print(coord.EarthLocation.get_site_names())
        raise ValueError('The site is not understood')

	###################################
	# Try reading start and end times
	###################################
    try:
        start_time = Time(start+' 12:01:00',location=observation_site)
        end_time = Time(end+' 12:01:00',location=observation_site)
        number_of_nights = int(end_time.jd - start_time.jd)
        time_range = Time([start+' 12:01:00',end+' 12:01:00'])
        print('Number of nights: {}'.format(number_of_nights))
    except:
        raise ValueError('Start and end times not understood')


	#####################
	# Now do constraints
	#####################
	#try:

    constraints = [AltitudeConstraint(0*u.deg, 90*u.deg),AirmassConstraint(3), AtNightConstraint.twilight_civil()]
	#except:
	#	raise ValueError('Unable to get set constraints')
	

    if method=='by_night':
        for i in range(number_of_nights):
            start_time_tmp = start_time  + TimeDelta(i,format='jd') #  get start time (doesent need to be accurate) 
            end_time_tmp = start_time  + TimeDelta(i+1,format='jd') #  get start time (doesent need to be accurate) 
            print('#'*80)	
            start_time_tmpss = start_time_tmp.datetime.ctime().split()	 # ['Fri', 'Dec', '24', '12:00:00', '2010']
            print('Night {} - {} {} {} {}'.format(i+1, start_time_tmpss[0],start_time_tmpss[2],start_time_tmpss[1],start_time_tmpss[-1]))
            print('#'*80)			


            # Now print Almnac information (sunset and end of evening twilight
            print('Almnac:')
            sun_set = observation_handle.sun_set_time(start_time_tmp,which='next')
            print('Sunset:\t\t\t\t\t\t\t'+sun_set.utc.datetime.ctime())

            twilight_evening_astronomical = observation_handle.twilight_evening_astronomical(start_time_tmp,which='next') # -18
            twilight_evening_nautical = observation_handle.twilight_evening_nautical(start_time_tmp,which='next') # -12
            twilight_evening_civil = observation_handle.twilight_evening_civil(start_time_tmp,which='next') # -6 deg
            print('Civil evening twilight (-6 deg) (U.T.C):\t\t'+twilight_evening_civil.utc.datetime.ctime())
            print('Nautical evening twilight (-12 deg) (U.T.C):\t\t'+twilight_evening_nautical.utc.datetime.ctime())
            print('Astronomical evening twilight (-18 deg) (U.T.C):\t'+twilight_evening_astronomical.utc.datetime.ctime())
            print('\n')

            twilight_morning_astronomical = observation_handle.twilight_morning_astronomical(start_time_tmp,which='next') # -18
            twilight_morning_nautical = observation_handle.twilight_morning_nautical(start_time_tmp,which='next') # -12
            twilight_morning_civil = observation_handle.twilight_morning_civil(start_time_tmp,which='next') # -6 deg
            print('Astronomical morning twilight (-18 deg) (U.T.C):\t'+twilight_morning_astronomical.utc.datetime.ctime())
            print('Nautical morning twilight (-12 deg) (U.T.C):\t\t'+twilight_morning_nautical.utc.datetime.ctime())
            print('Civil morning twilight (-6 deg) (U.T.C):\t\t'+twilight_morning_civil.utc.datetime.ctime())
            sun_rise = observation_handle.sun_rise_time(start_time_tmp,which='next')
            print('Sunrise:\t\t\t\t\t\t'+sun_rise.utc.datetime.ctime())
            print('\n')




            # stuff for creating plot
            plot_mids = []
            plot_names = []
            plot_widths = []


            for j in range(len(target_table)):
                # Extract information
                star_coordinates = coord.SkyCoord('{} {}'.format(target_table['RA'][j], target_table['DEC'][j]), unit=(u.hourangle, u.deg), frame='icrs')
                star_fixed_coord = FixedTarget(coord=star_coordinates,name=target_table['target'][j])


                ####################
                # Get finder image
                ####################
                '''
                plt.close()
                try:
                finder_image = plot_finder_image(star_fixed_coord,reticle=True,fov_radius=10*u.arcmin)
                except:
                pass
                plt.savefig(target_table['target'][j]+'_finder_chart.eps')
                '''

                P = target_table['Period'][j]
                Secondary_transit = target_table['Secondary'][j]
                transit_half_width = TimeDelta(target_table['width'][j]*60*60/2,format='sec') # in seconds for a TimeDelta

	            # now convert T0 to HJD -> JD -> BJD so we can cout period
                if 'epoch_HJD-2400000' in target_table.colnames:
                    #print('Using HJD-2400000')
                    T0 = target_table['epoch_HJD-2400000'][j]
                    T0 = Time(T0+2400000,format='jd') # HJD given by WASP
                    ltt_helio = T0.light_travel_time(star_coordinates, 'heliocentric',location=observation_site)
                    T0 = T0 - ltt_helio # HJD -> JD
                    ltt_bary = T0.light_travel_time(star_coordinates, 'barycentric',location=observation_site)
                    T0 = T0 + ltt_bary # JD -> BJD
                elif 'epoch_BJD-2455000' in target_table.colnames:
                    #print('Using BJD-2455000')
                    T0 = target_table['epoch_BJD-2455000'][j] + 2455000
                    T0 = Time(T0,format='jd') # BJD
                else:
                    print('\n\n\n\n FAILE\n\n\n\n')
                    continue


                ##########################################################
                # Now start from T0 and count in periods to find transits
                ##########################################################
                # convert star and end time to BJD
                ltt_bary_start_time = start_time_tmp.light_travel_time(star_coordinates, 'barycentric',location=observation_site)# + TimeDelta(i,format='jd')
                start_time_bary = start_time_tmp + ltt_bary_start_time # + TimeDelta(i,format='jd') #  convert start time to BJD

                ltt_bary_end_time_tmp = end_time_tmp.light_travel_time(star_coordinates, 'barycentric',location=observation_site)# + TimeDelta(i,format='jd')
                end_time_bary = end_time_tmp + ltt_bary_start_time  #+ TimeDelta(i+1,format='jd') #  convert end time to BJD and add 1 day 12pm -> 12pm the next day

                elapsed = end_time_bary - start_time_bary # now this is 24 hours from the start day 12:00 pm


                # now count transits
                time = Time(T0.jd,format='jd') # make a temporary copy
                transits = []
                primary_count, secondary_count = 0,0
                while time.jd < end_time_bary.jd:
                    if (time.jd>start_time_bary.jd) and (time.jd<end_time_bary.jd):
                        if is_observable(constraints, observation_handle, [star_fixed_coord], times=[time])[0] == True:
                            transits.append(time)
                            primary_count +=1
                    if Secondary_transit=='yes':
                        timesecondary = time + TimeDelta(P/2,format = 'jd')
                        if (timesecondary.jd>start_time_bary.jd) and (timesecondary.jd<end_time_bary.jd):
                            if is_observable(constraints, observation_handle, [star_fixed_coord], times=[timesecondary])[0] == True:
                                transits.append(timesecondary)
                                secondary_count +=1

                    time = time + TimeDelta(P,format = 'jd') # add another P to T0

                # Now find visible transits
                transits = [i for i in transits if is_observable(constraints, observation_handle, [star_fixed_coord], times=[i])[0] == True ]

				                

                if len(transits) == 0:
                    message = '{} has no transits.'.format(target_table['target'][j])
                    print('-'*len(message))			
                    print(message)
                    print('-'*len(message))			
                    print('\n')	
                    plt.close()
                    continue
                else:
                    message = '{} has {} primary transits and {} secondary transits.'.format(target_table['target'][j],primary_count, secondary_count)
                    print('-'*len(message))			
                    print(message)
                    print('RA: {}'.format(target_table['RA'][j]))
                    print('DEC: {}'.format(target_table['DEC'][j]))
                    print('Epoch: 2000')
                    print('T0 (BJD): {}'.format(T0.jd))
                    print('Period: {}'.format(P))
                    print('Transit width (hr): {}'.format(target_table['width'][j]))
                    print('-'*len(message))			
                    print('\n')	

                for i in transits:
                    # currently transit times are in BJD (need to convert to HJD to check
                    ltt_helio = i.light_travel_time(star_coordinates, 'barycentric',location=observation_site)
                    ii = i-ltt_helio
                    ltt_helio = ii.light_travel_time(star_coordinates, 'heliocentric',location=observation_site)
                    ii = ii+ltt_helio

                    transit_1 = i - transit_half_width - TimeDelta(7200,format='sec') # ingress - 2 hr
                    transit_2 = i - transit_half_width - TimeDelta(3600,format='sec') # ingress - 2 hr
                    transit_3 = i - transit_half_width  # ingress
                    transit_4 = i + transit_half_width  # egress 
                    transit_5 = i + transit_half_width + TimeDelta(3600,format='sec') # ingress - 2 hr
                    transit_6 = i + transit_half_width + TimeDelta(7200,format='sec') # ingress - 2 hr 

                    if (((i.jd-time.jd)/P) - np.floor((i.jd-time.jd)/P) < 0.1) or (((i.jd-time.jd)/P) - np.floor((i.jd-time.jd)/P) > 0.9):
                        print('Primary Transit:')	
                        print('-'*len('Primary Transit'))
                    if 0.4<((i.jd-time.jd)/P) - np.floor((i.jd-time.jd)/P) < 0.6:
                        print('Secondary Transit')
                        print('-'*len('Secondary Transit'))

                    ##################
                    # now get sirmass
                    ##################
                    altaz = star_coordinates.transform_to(AltAz(obstime=transit_1,location=observation_site))
                    hourangle = observation_handle1.target_hour_angle(transit_1,star_coordinates);hourangle = 24*hourangle.degree/360
                    if hourangle > 12:
                        hourangle -= 24
                    print('Ingress - 2hr (U.T.C):\t\t\t\t\t'+ transit_1.utc.datetime.ctime()+'\tAirmass: {:.2f}\tHA:{:.2f}'.format(altaz.secz,hourangle))

                    altaz = star_coordinates.transform_to(AltAz(obstime=transit_2,location=observation_site))
                    hourangle = observation_handle1.target_hour_angle(transit_2,star_coordinates);hourangle = 24*hourangle.degree/360
                    if hourangle > 12:
                        hourangle -= 24
                    print('Ingress - 1hr (U.T.C):\t\t\t\t\t'+ transit_2.utc.datetime.ctime()+'\tAirmass: {:.2f}\tHA:{:.2f}'.format(altaz.secz,hourangle))

                    altaz = star_coordinates.transform_to(AltAz(obstime=transit_3,location=observation_site))
                    hourangle = observation_handle1.target_hour_angle(transit_3,star_coordinates);hourangle = 24*hourangle.degree/360
                    if hourangle > 12:
                        hourangle -= 24
                    print('Ingress (U.T.C):\t\t\t\t\t'+ transit_3.utc.datetime.ctime()+'\tAirmass: {:.2f}\tHA:{:.2f}'.format(altaz.secz,hourangle))

                    altaz = star_coordinates.transform_to(AltAz(obstime=i,location=observation_site))
                    hourangle = observation_handle1.target_hour_angle(i,star_coordinates);hourangle = 24*hourangle.degree/360
                    if hourangle > 12:
                        hourangle -= 24
                    print('Mid transit (U.T.C):\t\t\t\t\t'+ i.utc.datetime.ctime()+'\tAirmass: {:.2f}\tHA:{:.2f}'.format(altaz.secz,hourangle))

                    altaz = star_coordinates.transform_to(AltAz(obstime=transit_4,location=observation_site))
                    hourangle = observation_handle1.target_hour_angle(transit_4,star_coordinates);hourangle = 24*hourangle.degree/360
                    if hourangle > 12:
                        hourangle -= 24
                    print('Egress (U.T.C):\t\t\t\t\t\t'+ transit_4.utc.datetime.ctime()+'\tAirmass: {:.2f}\tHA:{:.2f}'.format(altaz.secz,hourangle))

                    altaz = star_coordinates.transform_to(AltAz(obstime=transit_5,location=observation_site))
                    hourangle = observation_handle1.target_hour_angle(transit_5,star_coordinates);hourangle = 24*hourangle.degree/360
                    if hourangle > 12:
                        hourangle -= 24
                    print('Egress + 1hr (U.T.C):\t\t\t\t\t'+ transit_5.utc.datetime.ctime()+'\tAirmass: {:.2f}\tHA:{:.2f}'.format(altaz.secz,hourangle))

                    altaz = star_coordinates.transform_to(AltAz(obstime=transit_6,location=observation_site))
                    hourangle = observation_handle1.target_hour_angle(transit_6,star_coordinates);hourangle = 24*hourangle.degree/360
                    if hourangle > 12:
                        hourangle -= 24
                    print('Egress + 2hr (U.T.C):\t\t\t\t\t'+ transit_6.utc.datetime.ctime()+'\tAirmass: {:.2f}\tHA:{:.2f}'.format(altaz.secz,hourangle))
                    print('HJD {} (to check with http://var2.astro.cz/)\n'.format(ii.jd))


                    # append stuff for plots
                    plot_mids.append(i) # astropy Time
                    plot_names.append(target_table['target'][j])
                    plot_widths.append(target_table['width'][j])

            # Now plot 

            plt.close()
            if len(plot_mids) == 0:
                continue
            date_formatter = dates.DateFormatter('%H:%M')
            #ax.xaxis.set_major_formatter(date_formatter)




            # now load dummy transit lightcurves
            xp,yp = np.load('lc.npy')
            xs,ys = np.load('lcs.npy')

            # x = np.linspace(0, 2*np.pi, 400)
            # y = np.sin(x**2)

            subplots_adjust(hspace=0.000)
            number_of_subplots=len(plot_names) # number of targets transiting that night

            time = sun_set + np.linspace(-1,14,100)*u.hour # take us to sunset
            for i,v in enumerate(xrange(number_of_subplots)):
                # exctract params
                width = plot_widths[v]
                name = plot_names[v]
                mid = plot_mids[v]

                # now set up dummy lc plot
                x_tmp =mid + xp*(width/2)*u.hour # get right width in hours



                # now set up axis
                v = v+1
                ax1 = subplot(number_of_subplots,1,v)
                ax1.xaxis.set_major_formatter(date_formatter)

                if v == 1:
                    ax1.set_title(start)


                # plot transit model
                ax1.plot_date(x_tmp.plot_date, ys,'k-')

                # plot continuum
                #xx  =time.plot_date
                #xx = [uu for uu in xx if (uu<min(x_tmp.plot_date)) or (uu>max(x_tmp.plot_date))]

                #ax1.plot_date(xx, np.ones(len(xx)),'k--', alpha=0.3)
                ax1.set_xlim(min(time.plot_date),max(time.plot_date))
                #ax1.plot_date(mid.plot_date, 0.5, 'ro')
                plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')
                ax1.set_ylabel(name,rotation=45,labelpad=20)


                twilights = [
                (sun_set.datetime, 0.0),
                (twilight_evening_civil.datetime, 0.1),
                (twilight_evening_nautical.datetime, 0.2),
                (twilight_evening_astronomical.datetime, 0.3),
                (twilight_morning_astronomical.datetime, 0.4),
                (twilight_morning_nautical.datetime, 0.3),
                (twilight_morning_civil.datetime, 0.2),
                (sun_rise.datetime, 0.1),
                ]


                for ii, twii in enumerate(twilights[1:], 1):
                    ax1.axvspan(twilights[ii - 1][0], twilights[ii][0], ymin=0, ymax=1, color='grey', alpha=twii[1])

                ax1.grid(alpha=0.5)
                ax1.get_yaxis().set_ticks([])
                if v != number_of_subplots:
                    ax1.get_xaxis().set_ticks([])
				

            plt.xlabel('Time [U.T.C]')
            #plt.tight_layout()
            #plt.savefig('test.eps',format='eps')
            plt.show()


	
'''
    # Shade background during night time
    if brightness_shading:
        start = time[0].datetime

        # Calculate and order twilights and set plotting alpha for each
        twilights = [
            (observer.sun_set_time(Time(start), which='next').datetime, 0.0),
            (observer.twilight_evening_civil(Time(start), which='next').datetime, 0.1),
            (observer.twilight_evening_nautical(Time(start), which='next').datetime, 0.2),
            (observer.twilight_evening_astronomical(Time(start), which='next').datetime, 0.3),
            (observer.twilight_morning_astronomical(Time(start), which='next').datetime, 0.4),
            (observer.twilight_morning_nautical(Time(start), which='next').datetime, 0.3),
            (observer.twilight_morning_civil(Time(start), which='next').datetime, 0.2),
            (observer.sun_rise_time(Time(start), which='next').datetime, 0.1),
        ]

        for i, twi in enumerate(twilights[1:], 1):
            ax.axvspan(twilights[i - 1][0], twilights[i][0], ymin=0, ymax=1, color='grey', alpha=twi[1])




time = time + np.linspace(-12, 12, 100)*u.hour
ax.plot_date(time.plot_date, masked_airmass, label=target_name, **style_kwargs)

    # Invert y-axis and set limits.
    if ax.get_ylim()[1] > ax.get_ylim()[0]:
        ax.invert_yaxis()
    ax.set_ylim([3, 1])
    ax.set_xlim([time[0].plot_date, time[-1].plot_date])

    # Set labels.
    ax.set_ylabel("Airmass")
    ax.set_xlabel("Time from {0} [UTC]".format(min(time).datetime.date()))

    if altitude_yaxis and not _has_twin(ax):
        altitude_ticks = np.array([90, 60, 50, 40, 30, 20])
        airmass_ticks = 1./np.cos(np.radians(90 - altitude_ticks))

        ax2 = ax.twinx()
        ax2.invert_yaxis()
        ax2.set_yticks(airmass_ticks)
        ax2.set_yticklabels(altitude_ticks)
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Altitude [degrees]')

    # Redraw figure for interactive sessions.
    ax.figure.canvas.draw()
'''			

				


					
						

						
