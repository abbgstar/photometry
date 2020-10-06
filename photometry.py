import numpy as np
from astropy.io import fits
from astropy import wcs
import sep
from array_io import *
import os, sys, argparse
import time
from helper_scripts import *
import scipy.ndimage as nd

"""
Example:
    python photometry.py mosaics psfs outfiles F200W.global_rms.64.det.cat F200W.mask.fits -p F200W -m 0 -w 64 -s -i
"""

def make_objects(n):
    objects = np.zeros(n, dtype=[('npix','i8'),('x','f8'),('y','f8'),('a','f8'),('b','f8'),('theta','f8'),('ido','i8'), ('kronrad','f8'),('krflag','i8')])
    return objects

def main():
    all_start = time.time()

    #Handle user input with argparse
    parser = argparse.ArgumentParser(description="photometry flags and options from user")    

    parser.add_argument('mosaic_dir',
	                    metavar = 'path to mosaics',
	                    type=str,
	                    help='directory containing mosaics')

    parser.add_argument('psf_dir',
                        metavar = 'path to psf files',
                        type=str,
                        help='directory containing psf files')

    parser.add_argument('out_dir',
                        metavar = 'path to output files',
                        type=str,
                        help='directory to hold output files')
                        
    parser.add_argument('source_cat',
                        metavar = 'name of source catalog',
                        type=str,
                        help='name of source catalog')

    parser.add_argument('source_mask',
                        metavar = 'mask used in generating source catalog',
                        type=str,
                        help='name of source mask')

    parser.add_argument('-p',
    		            '--phot_band',
    		            metavar='photometry band',
    		            type=str,
    		            help='Valid JWST band',
    		            choices=['F090W' , 'F115W', 'F150W', 'F200W', 'F277W' , 'F335M', 'F356W', 'F410M', 'F444W'],
    		            required=True,
    		            default = 'F200W')

    parser.add_argument('-m',
    		            '--flag_mode',
    		            metavar='flag mode',
    		            type=int,
    		            help='Method to use as error for extraction. global background rms (0) or background rms image (1) or data error image (2)',
    		            choices=[0,1,2],
    		            default=0)

    parser.add_argument('-w',
                        '--window_size',
    		            metavar = 'window size',
    		            type=int,
    		            help='Window size for error bkg calc',
    		            default=64)

    parser.add_argument('-s',
    		            '--flag_sort',
    		            action='store_false',
    		            help='keep sort order from input catalog',
    		            default=True)

    parser.add_argument('-i',
    		            '--flag_ids',
    		            action='store_true',
    		            help='keep ids from input catalog',
    		            default=False)

    parser.add_argument('-psf',
    		            '--flag_calc_psf',
    		            action='store_true',
    		            help='Perform psf calculation',
    		            default=False) 

    args = parser.parse_args()

    prefix_path = os.getcwd() + os.sep

    #display extraction method to be used, and generate string to append to catalog name to reflect flag_mode
    if(args.flag_mode==0):
    	print("Using global background rms as error for extraction.")
    	catstr = '.global_rms.'
    if(args.flag_mode==1):
    	print("Using background rms image as error for extraction.")
    	catstr = '.bkgrd_rms.'
    if(args.flag_mode==2):
    	print("Using data error image as error for extraction.")
    	catstr = '.data_err.'

    print("New detection band: ",args.phot_band)

    #check validity of mosaics directory and append trailing slash
    if not os.path.isdir(prefix_path + args.mosaic_dir + os.sep):
    	print ('Mosaic path is not a valid directory')
    	sys.exit()

    fdir_mosaics = prefix_path + args.mosaic_dir + os.sep

    #check if outfiles directory exist, exit if it does not, and append trailing slash
    if not os.path.isdir(prefix_path + args.out_dir + os.sep):
    	print ('Outfiles path is not a valid directory -- will now exit')
    	sys.exit()
    else:
    	fdir_outfiles = prefix_path + args.out_dir + os.sep

    #construct some filenames with full paths
    catstr += str(args.window_size)   #string to uniquely identify window size and flag_mode

    fname_source_cat  = fdir_outfiles + args.source_cat
    fname_source_mask = fdir_outfiles + args.source_mask

    #read in mask
    fmask, header_mask = read_fits(fname_source_mask)
    idx_mask = np.where(np.isnan(fmask)==True)
    mask = np.empty(fmask.shape,dtype=bool)
    mask[idx_mask] = True

    #read in source catalog
    fp = open(fname_source_cat,"r")
    fl = fp.readlines()
    n_cat = len(fl)-1

    objects = make_objects(n_cat)
    for i in range(n_cat):
        l = fl[i+1]
        objects['ido'][i] = int(l.split()[0])
        objects['x'][i] = float(l.split()[1])
        objects['y'][i] = float(l.split()[2])
        objects['a'][i] = float(l.split()[5])
        objects['b'][i] = float(l.split()[6])
        objects['theta'][i] = float(l.split()[7])
        objects['npix'][i] = int(l.split()[8])
        objects['kronrad'][i] = float(l.split()[9])   #drop this in write_sep_cat
        objects['krflag'][i] = int(l.split()[10])     #drop this in write_sep_cat
    fp.close()
    
    fname_data = fdir_mosaics + args.phot_band + ".fits"
    fname_err = fdir_mosaics + args.phot_band + ".err.fits"

    #read mosaicked data and errors
    data, header_data = read_fits(fname_data)
    data_err, header_err = read_fits(fname_err)

    #read in the wcs
    wcs_obj = wcs.WCS(header_data)

    flux_conv = 10**(-0.4 *( header_data['ABMAG'] - 31.4 ) )
    print("nJy zeropoint = ",flux_conv)

    #get indices to real and NaN values
    idx_real = np.where( (np.isnan(data)==False)&(np.isnan(data_err)==False) )
    idx_nan = np.where( (np.isnan(data)==True)&(np.isnan(data_err)==True) )
    int_nan = -99999

    # measure a spatially varying background on the image
    bkg = sep.Background(data, mask=mask, bw=args.window_size, bh=args.window_size)

    #get the background image
    bkg_image = bkg.back()
    bkg_rms= bkg.rms()
    bkg_image[idx_nan] = np.nan

    #subtract the background
    data_sub = data - bkg

    #save bkg subtracted image
    fname_sub = "source_det_" + args.phot_band + ".sub.fits"
    print("Writing background subtracted ",fname_sub)
    fits.writeto(fdir_outfiles + fname_sub, data_sub, header_data, overwrite=True)

    ### determine the EE PSF aperture
    t_start = time.time()

    if(args.flag_calc_psf):

	    if not os.path.isdir(prefix_path + args.psf_dir + os.sep):
	    	print ('psf path is not a valid directory')
	    	sys.exit()
	    fname_psf = fdir_psfs + "PSF_" + args.det_band + ".fits"
	    print("Reading PSF ",fname_psf)
	    hdul_psf = fits.open(fname_psf)
	    data_psf = hdul_psf[0].data
	    header_psf = hdul_psf[0].header
	    psf_pixel_scale  = header_psf['PIXELSCL']  #Scale in arcsec/pix (after oversampling)

	    xi = np.zeros(data_psf.shape,dtype=int)
	    yi = np.zeros(data_psf.shape,dtype=int)
	    #print(xi.shape)
	    for i in range(data_psf.shape[0]):
        		xi[i,:] = i
	    for j in range(data_psf.shape[1]):
        		yi[:,j] = j

	    xic = int(0.5*data_psf.shape[0])
	    yic = int(0.5*data_psf.shape[1]) 
	    print(xic,yic,data_psf.shape[0],data_psf.shape[1])
	    idx_max = np.where(data_psf==data_psf.max())
	    #print(
	    xic = int(xi[idx_max])
	    yic = int(yi[idx_max])
	    ri = ( (xi-xic)**2 + (yi-yic)**2 )**0.5

        # dumb search for r_psf_70
	    flag = True
	    rtest = 10.0
	    tol = 5.0e-3
	    iter = 0
	    while(flag):
	    	idx = np.where(ri<rtest)
	    	flux = np.sum(data_psf[xi[idx],yi[idx]])
	    	if(np.abs(flux-0.7)>tol):
	    		if(flux<0.7):
	    			rtest*=1.1#*(tol/flux)
            		#rtest=rtest**1.1#*(tol/flux)
	    		else:
	    			rtest/=1.05
	    			#rtest=rtest**(-1.05)
	    	else:
	    		flag = False
        	#print(rtest,flux,np.abs(flux-0.7),tol)
	    	iter+=1.
	    	if(iter>100):
	    		print("Error converging in PSF search.")
	    		flag = 0
	    	#print(rtest,flux)
	    arcsec_per_degree = 3600.
	    pixel_scale = header_data['CDELT2'] * arcsec_per_degree
	    r_psf_70 = rtest*psf_pixel_scale/pixel_scale
	    s = "Radius enclosing %4.2f PSF flux = %f (rt = %f) " % (flux,r_psf_70,rtest*psf_pixel_scale)
	    #print("Radius enclosing 70% PSF flux = ",r_psf_70)
	    print(s)
	    r_psf = r_psf_70

    else:
	    #80% enclosed
	    rad_dict = {"F090W":0.232, "F115W":0.213, "F150W":0.189, "F200W":0.176, "F277W":0.195, "F335M":0.225, "F356W":0.235, "F410M":0.266, "F444W":0.283}
	    arcsec_per_degree = 3600.
	    pixel_scale = header_data['CDELT2'] * arcsec_per_degree
	    rtest = rad_dict[args.phot_band]
	    r_psf_80 = rtest  / pixel_scale 
	    print("Radius enclosing 80% PSF flux = ",r_psf_80,)
	    r_psf = r_psf_80

    t_end = time.time()
    print("EE percent flux radius time elapsed = ",t_end-t_start)

    ### Compute FLUX in radius containing EE of the PSF flux
    x = objects['x']
    y = objects['y']

    if(args.flag_mode==0):
    	cflux_r_psf, cfluxerr_r_psf, cflag_r_psf= sep.sum_circle(data_sub, x, y,
            	r_psf, mask=mask, err=bkg.globalrms, subpix=1)
    if(args.flag_mode==1):
    	cflux_r_psf, cfluxerr_r_psf, cflag_r_psf= sep.sum_circle(data_sub, x, y,
             	r_psf, mask=mask, err=bkg_rms, subpix=1)
    if(args.flag_mode==2):
    	cflux_r_psf, cfluxerr_r_psf, cflag_r_psf= sep.sum_circle(data_sub, x, y,
            	r_psf, mask=mask, err=data_err, subpix=1)

    # ### Compute random aperture flux
    n_random = 100000
    x_random = np.random.random(n_random)*data.shape[0]
    y_random = np.random.random(n_random)*data.shape[1]
    r_random = r_psf
    mask_sig = np.empty_like(data,dtype=bool)
    idx_sig = np.where(np.abs(data_sub/data_err)>3.0)
    mask_sig[idx_sig] = True
    mask_sig[idx_nan] = True
    mask_sig[idx_mask] = True

    if(args.flag_mode==0):
    	cflux_random, cfluxerr_random, cflag_random = sep.sum_circle(data_sub, x_random, y_random,
            	r_random, mask=mask_sig, err=bkg.globalrms, subpix=1)
    if(args.flag_mode==1):
    	cflux_random, cfluxerr_random, cflag_random = sep.sum_circle(data_sub, x_random, y_random,
            	r_random, mask=mask_sig, err=bkg_rms, subpix=1)
    if(args.flag_mode==2):
    	cflux_random, cfluxerr_random, cflag_random = sep.sum_circle(data_sub, x_random, y_random,
            	r_random, mask=mask_sig, err=data_err, subpix=1)

    idx_ok = np.where(np.isnan(cflux_random)==False)
    sigma_r_psf= np.std(cflux_random[idx_ok])
    print("RMS flux in EE PSF radius = ",sigma_r_psf," [counts]")
    print("RMS flux in EE PSF radius = ",flux_conv*sigma_r_psf," [nJy]")


    ### Compute FLUX_AUTO equivalent

    #compute kron radius, calculate flux_auto
    x = objects['x']
    y = objects['y']
    a = objects['a']
    b = objects['b']
    theta = objects['theta']
    theta = np.clip(theta,-0.5*np.pi,0.5*np.pi)

    #perform kron phot
    kronrad, krflag = sep.kron_radius(data_sub, x, y, a, b, theta, 6., mask=mask)
    ok_flag = np.where(np.isnan(kronrad)==False)

    flux = np.empty(len(x))
    fluxerr = np.empty(len(x))
    flag = np.empty(len(x),dtype=int)

    if(args.flag_mode==0):
	    flux[ok_flag], fluxerr[ok_flag], flag[ok_flag] = sep.sum_ellipse(data_sub,  x[ok_flag], y[ok_flag], a[ok_flag], b[ok_flag], theta[ok_flag],  2.5*kronrad[ok_flag], mask=mask, err=bkg.globalrms,subpix=1)
    if(args.flag_mode==1):
    	flux[ok_flag], fluxerr[ok_flag], flag[ok_flag] = sep.sum_ellipse(data_sub,  x[ok_flag], y[ok_flag], a[ok_flag], b[ok_flag], theta[ok_flag],  2.5*kronrad[ok_flag], mask=mask, err=bkg_rms,subpix=1)
    if(args.flag_mode==2):
    	flux[ok_flag], fluxerr[ok_flag], flag[ok_flag] = sep.sum_ellipse(data_sub,  x[ok_flag], y[ok_flag], a[ok_flag], b[ok_flag], theta[ok_flag],  2.5*kronrad[ok_flag], mask=mask, err=data_err,subpix=1)

    flag |= krflag  # combine flags into 'flag'

    #limit to a minimum diameter
    r_min = r_psf# minimum diameter = 2x r_psf

    use_circle = np.where( (kronrad * np.sqrt(a * b) < r_min) | (np.isnan(kronrad)==True))
    if(args.flag_mode==0):
    	cflux, cfluxerr, cflag = sep.sum_circle(data_sub, x, y, r_min, mask=mask, err=bkg.globalrms, subpix=1)
    if(args.flag_mode==1):
    	cflux, cfluxerr, cflag = sep.sum_circle(data_sub, x, y, r_min, mask=mask, err=bkg_rms, subpix=1)
    if(args.flag_mode==2):
    	cflux, cfluxerr, cflag = sep.sum_circle(data_sub, x, y, r_min, mask=mask, err=data_err, subpix=1)

    #print(use_circle)
    flux[use_circle] = cflux[use_circle]
    fluxerr[use_circle] = cfluxerr[use_circle]
    flag[use_circle] = cflag[use_circle]
    kronrad[use_circle] = r_min


    #output the catalog
    #find the ra and dec of sources
    x = objects['x']
    y = objects['y']
    pos = np.zeros((len(x),2))
    pos[:,0] = x
    pos[:,1] = y
    pos_sky = wcs_obj.wcs_pix2world(pos,0)

    catstr += ".phot.cat"
    fout = fdir_outfiles + args.phot_band + catstr
    print("Outputting the catalog as ",fout)

    write_phot_catalog(fout,args.phot_band,objects,pos_sky,kronrad,flux*flux_conv,fluxerr*flux_conv,flag,cflux*flux_conv,cfluxerr*flux_conv,cflag,args.flag_sort,args.flag_ids)
    #write_sep_catalog(fout,det_band,objects,pos_sky,kronrad,flux,fluxerr,flag,cflux,cfluxerr,cflag,flag_sort,flag_ids)

    all_end = time.time()
    print("Total time elapsed = ",all_end-all_start)
    print("Done!")

if __name__ == "__main__": main()  