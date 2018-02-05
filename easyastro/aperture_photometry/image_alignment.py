import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import autojit

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

@autojit(nopython=True)
def count_flux(image, x,y,r1,r2,r3):
    flux1 = 0.
    flux2 = 0.
    flux3 = 0.
    tmp1 = 0.0
    r1 = r1*r1
    r2 = r2*r2
    r3 = r3*r3

    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            tmp1 = i*i + j*j
            if (tmp1 < r1):
                flux1 = flux1 + image[j,i]
            if (tmp1 < r2):
                flux2 = flux2 + image[j,i]
            if (tmp1 < r3):
                flux3 = flux3 + image[j,i]

    return flux1,flux2,flux3





def find_offset(im1, im2, xlim, ylim , method = 'gaussian'):
    ############################
    # Note original shapes
    ############################
    orig_shape1 = np.array(im1.shape)
    orig_shape2 = np.array(im2.shape)

    #####################
    # Preliminary checks
    #####################
    if xlim[0] > xlim[1]:
        raise ValueError('The first x limit cannot exceed the second.')
    if ylim[0] > ylim[1]:
        raise ValueError('The first y limit cannot exceed the second.')

    ####################
    # First trim images
    ####################
    im1 = im1[xlim[0]:xlim[1], ylim[0]:ylim[1] ]
    im2 = im2[xlim[0]:xlim[1], ylim[0]:ylim[1] ]

    ##############################################################
    # get rid of the averages, otherwise the results are not good
    ##############################################################
    im1 -= np.mean(im1)
    im2 -= np.mean(im2)

    ##########################################################################
    # calculate the correlation image; note the flipping of onw of the images
    ##########################################################################
    corr_img =  fftconvolve(im1, im2[::-1,::-1], mode='same')


    if method == 'best_pixel':
        ##############################################################################
        # Now unravel the offset by finding the peak
        # Note that this is relative to ratio of the original image and the cut image
        ###############################################################################
        best_corr = np.array(np.unravel_index(np.argmax(corr_img), corr_img.shape)) # the best coordinates
        dx, dy = (best_corr[::-1] - np.array(corr_img.shape)[::-1]/2) * np.array([-1,-1])
        dxe, dye = 0.5,  0.5
        #return (best_corr[::-1] - np.array(corr_img.shape)[::-1]/2) * np.array([-1,-1])
        return dx,dxe,dy,dye

    elif method == 'gaussian_fit':
        ######################################################
        # Get the mean of the CCF image in y and x direction
        ######################################################
        corr_img_y , corr_img_x = corr_img.mean(axis=0), corr_img.mean(axis=1)

        corr_img_x  =corr_img_x - np.min(corr_img_x)
        corr_img_x = corr_img_x/ corr_img_x.max()

        corr_img_y  =corr_img_y - np.min(corr_img_y)
        corr_img_y = corr_img_y / corr_img_y.max()
        

        #######################################
        # Now get the pixel it socrresponds to
        #########################################
        corr_img_yy , corr_img_xx = np.arange(corr_img.shape[1]) - np.array(corr_img.shape[1])/2  ,  np.arange(corr_img.shape[0]) - np.array(corr_img.shape[0])/2 


	
        #####################
        # do y first
        #####################  
        n = corr_img_xx.shape[0]
        n_low, n_high = np.int(np.floor(0.35*n)), np.int(np.floor(0.7*n))
        corr_img_x = corr_img_x[n_low:n_high]
        corr_img_xx = corr_img_xx[n_low:n_high]

        mean = np.sum(corr_img_xx* corr_img_x) / np.sum(corr_img_x)
        sigma = np.sqrt(np.sum(corr_img_x * (corr_img_xx - mean)**2) / np.sum(corr_img_x))
        try:
            popt,pcov = curve_fit(Gauss, corr_img_xx, corr_img_x, p0=[max(corr_img_x), 0, 1])
            perr = np.sqrt(np.diag(pcov))
            dx, dxe = -popt[1], perr[1]
        except:
            dx,dxe = 0,99

        '''
        plt.close()
        plt.plot(corr_img_xx,Gauss(corr_img_xx,*popt),'r:',label='fit')
        plt.plot(corr_img_xx,corr_img_x,'b')
        plt.show()
        plt.sleep(1)
        '''



        #####################
        # do x next
        ##################### 
        n = corr_img_yy.shape[0]
        n_low, n_high = np.int(np.floor(0.35*n)), np.int(np.floor(0.7*n))
        corr_img_y = corr_img_y[n_low:n_high]
        corr_img_yy = corr_img_yy[n_low:n_high]
    
        mean = np.sum(corr_img_yy* corr_img_y) / np.sum(corr_img_y)
        sigma = np.sqrt(np.sum(corr_img_y * (corr_img_y - mean)**2) / np.sum(corr_img_y))
        try:
            popt,pcov = curve_fit(Gauss, corr_img_yy, corr_img_y, p0=[max(corr_img_y), 0, 1])
            perr = np.sqrt(np.diag(pcov))
            dy, dye = -popt[1], perr[1]
        except:
            dy,dye = 0,99

        '''
        plt.close()
        plt.plot(corr_img_yy,Gauss(corr_img_yy,*popt),'r:',label='fit')
        plt.plot(corr_img_yy,corr_img_y,'b')
        plt.show()
        plt.sleep(1)
        '''
        ##################
        # Ooutput result
        ##################
        #print('dx: {:.3f} +/- {:.3f}, dy: {:.3f} +/- {:.3f}'.format(dx,dxe,dy,dye))

        return dy,dye,dx,dxe

    else:
        msg='''
Method choice not understood.

Available choices are:

1) best_pixel

2) gaussian_fit
'''
        raise ValueError(msg)

           
        
        

def time_series_photometry(reference_image='a8947442.fits', images='all', reference_star = [392,327], comparison_stars = [], radius_aper=5, radius_sky=10, radius_annulus=15, xlim=[302,350], ylim = [373,423], alignment_method='gaussian_fit', time_header = 'MJD-OBS', other_headers=[]):   
    if images=='all':
        images = glob.glob('*.fits')
        images.sort()

    # load the reference image  
    reference_image = fits.open(reference_image)[0].data.astype(np.float)

    # initiate the array which will hold all of the information
    # filename,time, x, y, dx, dy, [flux1, flux2, flux3] *, other_headers *, 
    photometry_information = np.zeros((len(images), 9 + (3*len(comparison_stars) + 1) + len(other_headers) )).astype(object)

    plt.ion()
    for i in range(len(images)):
        plt.cla()
        # load the image
        image = fits.open(images[i])[0].data.astype(np.float)
        header = fits.open(images[i])[0].header

        # Now get time
        time = header[time_header] 

        # plot the image
        plt.imshow(image, origin='lower') 
        plt.ylim(xlim[0], xlim[1])
        plt.xlim(ylim[0], ylim[1])
        plt.xlabel('X - pixel')
        plt.ylabel('Y - pixel')
        plt.title(images[i][:-5])

        # find image offsett
        dx,dxe,dy,dye = find_offset(reference_image, image, xlim=xlim, ylim = ylim, method=alignment_method)

        # plot the apertures with offset
        circle1 = plt.Circle((reference_star[0] + dx, reference_star[1] + dy), radius_aper, color='r', fill=False)
        plt.gca().add_artist(circle1)

        circle2 = plt.Circle((reference_star[0] + dx, reference_star[1] + dy), radius_sky, color='r', fill=False)
        plt.gca().add_artist(circle2)

        circle3 = plt.Circle((reference_star[0] + dx, reference_star[1] + dy), radius_annulus, color='r', fill=False)
        plt.gca().add_artist(circle3) 

        # Now get the flux of target star
        flux1, flux2, flux3 = count_flux(image, reference_star[0] + dx, reference_star[1] + dy,radius_aper,radius_sky,radius_annulus)  

   

           
        # now append data so far
        # filename,time, x+dx, y+dy, dx, dy, [flux1, flux2, flux3] *, other_headers *,         
        photometry_information[i][0] = images[i]
        photometry_information[i][1] = time
        photometry_information[i][2] = reference_star[0] + dx
        photometry_information[i][3] = reference_star[1] + dy
        photometry_information[i][4] = dx
        photometry_information[i][5] = dy
        photometry_information[i][6],photometry_information[i][7],photometry_information[i][8] = flux1, flux2, flux3
            


        plt.draw()
        plt.pause(0.025)

    return photometry_information
         

        


import glob
files = glob.glob('a*.fits')
files.sort()

plt.ion()
best_coords = 0
for i in files[1:]:
    x,y = 392,327
    r1,r2,r3 = 10,20,30

    plt.cla()
    image_1 = fits.open(i)[0].data.astype(np.float)


    plt.imshow(image_1, origin='lower')

    dx,dxe,dy,dye = find_offset(reference_image, image_1, xlim=[302,350], ylim = [373,423], method='gaussian_fit')


    circle1 = plt.Circle((x + dx, y + dy), r1, color='r', alpha = 0.2, fill=False)
    plt.gca().add_artist(circle1)

    circle2 = plt.Circle((x + dx, y + dy), r2, color='r', alpha = 0.2, fill=False)
    plt.gca().add_artist(circle2)

    circle3 = plt.Circle((x + dx, y + dy), r3, color='r', alpha = 0.2, fill=False)
    plt.gca().add_artist(circle3)

    plt.plot(392 + dx, 327 + dy, 'r+')
    plt.draw()
    plt.ylim(302,350)
    plt.xlim(373,423)
    
    f1,f2,f3 = count_flux(image_1, x + dx, y + dy,r1,r2,r3) 

    print('{}   dx: {:.3f} +/- {:.3f}      dy: {:.3f} +/- {:.3f}     {}       {}       {}'.format(i, dx,dxe,dy,dye,f1,f2,f3))

  
    #print(f1,f2,f3)
    plt.pause(0.1)
    
	# 392, 327


