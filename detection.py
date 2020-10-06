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
Input: 
        Detection image and associated error image
output:
        A SEP derived catalog of sourses with Kron radii and flags

Example:
        python detection.py detection outfiles -n F200W -x 10 -m 0 -w 64 -t 1.5 -dc 0.01 -s
        python detection.py -h for more details
"""

def main():
    all_start = time.time()

    #Handle user input with argparse
    parser = argparse.ArgumentParser(description="detection flags and options from user")

    parser.add_argument('detection_dir',
    		    metavar = 'detection image directory',
    		    type=str,
    		    help='directory containing detection image')

    parser.add_argument('out_dir',
    		    metavar = 'output files directory',
    		    type=str,
    		    help='directory to hold output files')

    parser.add_argument('-n',
    		    '--name',
    		    metavar='name of detection image',
    		    type=str,
    		    help='name of detection image (without .fits extension)',
    		    required=True,
    		    default = 'F200W')

    parser.add_argument('-x',
    		    '--flag_mask',
    		    metavar = 'flag mask',
    		    type=int,
    		    help='flag mask',
    		    required=True,
    		    default = 10)

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

    parser.add_argument('-t',
                '--thresh',
    	        metavar = 'threshold for SEP extraction (in sigma units)',
    	        type=float,
    	        help='threshold for SEP extraction (in sigma units)',
    	        default=1.5)

    parser.add_argument('-dc',
                '--deblend_contrast',
    	        metavar = 'deblend constrast for SEP extraction',
    	        type=float,
    	        help='deblend contrast for SEP extraction',
    	        default=0.01)

    parser.add_argument('-s',
		        '--flag_sort',
		        action='store_true',
		        help='sort extractions by kron radius',
		        default=False)

    args = parser.parse_args()
    prefix_path = os.getcwd() + os.sep

    #set nred (number of pixels to mask along image edge?)
    if(bool(int(args.flag_mask)) & (int(args.flag_mask)>1)):
    	nred = int(args.flag_mask) 

    #check validity of detection directory and append trailing slash
    if not os.path.isdir(prefix_path + args.detection_dir + os.sep):
    	print ('Detection path is not a valid directory')
    	sys.exit()

    fdir_detection = prefix_path + args.detection_dir + os.sep

    #check if outfiles directory exist, create a new one if it does not, and append trailing slash
    if not os.path.isdir(prefix_path + args.out_dir + os.sep):
    	print ('Outfiles path is not a valid directory -- will create one')
    	os.makedirs('outfiles')
    	fdir_outfiles = prefix_path + 'outfiles' + os.sep
    else:
    	fdir_outfiles = prefix_path + args.out_dir + os.sep
    
    print("Beginning detection...")

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
  
    fname_data = fdir_detection + args.name + ".fits"
    fname_err = fdir_detection + args.name + ".err.fits"

    #read mosaicked data and errors
    data, header_data = read_fits(fname_data)
    data_err, header_err = read_fits(fname_err)

    #read in the wcs
    wcs_obj = wcs.WCS(header_data)

    # flux_conv = 10**(-0.4 *( header_data['ABMAG'] - 31.4 ) )
    # print("nJy zeropoint = ",flux_conv)

    #get indices of real and NaN values in data
    idx_real = np.where((np.isnan(data)==False)&(np.isnan(data_err)==False))
    idx_nan = np.where((np.isnan(data)==True)&(np.isnan(data_err)==True))
    int_nan = -99999

    t_start = time.time()
    
    s = "Computing mask... (flag = " + str(bool(args.flag_mask)) +", nred = " + str(nred) +")"
    print(s)

    # ### Make a mask of NaN pixels
    mask = np.empty_like(data,dtype=bool)
    mask[idx_nan] = True

    if(args.flag_mask):
        #grow the mask slightly inward to deal with edges of the map
        fmaskm = np.full(mask.shape,1,dtype=float)
        idx_mask = np.where(mask==True)
        fmaskm[idx_mask] = 0

        #median filter the mask
        fmask_med = nd.median_filter(fmaskm, size=20)
        idx_fmask_med = np.where(fmask_med==0)
        fmask_med[idx_fmask_med] = np.nan

        #roll the mask slightly try rolling mask_med
        mask_med = 0.0
        mask_med = np.empty(fmask_med.shape,dtype=bool)
        idx_med = np.where(np.isnan(fmask_med)==True)
        mask_med[idx_med] = True

        mask_med_prev = False
        mask_med_prev = mask_med.copy()

        for i in range(mask_med.shape[0]):
            for j in range(1,nred):
                mask_med[i,:] = (mask_med[i,:]|mask_med_prev[i-j,:])
                if(i+j<mask_med.shape[0]):
                    mask_med[i,:] = (mask_med[i,:]|mask_med_prev[i+j,:])
            
        for i in range(mask_med.shape[1]):
            for j in range(1,nred):
                mask_med[:,i] = (mask_med[:,i]|mask_med_prev[:,i-j])
                if(i+j<mask_med.shape[1]):
                    mask_med[:,i] = (mask_med[:,i]|mask_med_prev[:,i+j])

        #save the modified mask
        mask = mask_med

    froll = 0.0
    froll = np.full(mask.shape,1,dtype=float)
    idx_mask = np.where(mask==True)
    froll[idx_mask] = np.nan
    if(args.flag_mask):	
    	print("Fractional area lost to camfer = ", (float(len(idx_mask[0]))-float(len(idx_nan[0])) )/float(len(idx_nan[0])))
    
    fname_roll = args.name+".mask.fits"
    print("Writing mask ",fname_roll)

    #convert bool to int
    int_mask = mask.astype(int)
    fits.writeto(fdir_outfiles + fname_roll, int_mask, header_data, overwrite=True)

    #print(len(idx_mask[0]))
    t_end = time.time()
    print("Time elapsed to compute mask = ",t_end-t_start)

    ### Perform background subtraction

    # measure a spatially varying background on the image
    bkg = sep.Background(data, mask=mask, bw=args.window_size, bh=args.window_size)  

    #get the background image
    bkg_image = bkg.back()
    #print(len(idx_nan[0]))
    bkg_image[idx_nan] = np.nan

    #subtract the background
    data_sub = data - bkg

    #save bkg subtracted image
    fname_sub = args.name+".sub.fits"
    print("Writing background subtracted image -- ",fname_sub)
    fits.writeto(fdir_outfiles + fname_sub, data_sub, header_data, overwrite=True)

    ### Perform object detection

    #set the sep paramters
    thresh = args.thresh # in sigma
    deblend_cont = args.deblend_contrast #also try 0.005
    minarea = 5  # minimum area in pixels
    filter_type = 'matched'
    deblend_nthresh = 32
    clean = True
    clean_param = 1.0

    #perform the extraction

    #get the bkg rms image
    bkg_rms = bkg.rms()

    t_start = time.time()
    print("Extracting sources...")

    if(args.flag_mode==0):
    	objects, seg_map = sep.extract(data_sub, thresh, err=bkg.globalrms, mask=mask, \
    		minarea=minarea, filter_type=filter_type, \
    		deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, \
    		clean=clean, clean_param=clean_param, segmentation_map=True)

    if(args.flag_mode==1):
    	objects, seg_map = sep.extract(data_sub, thresh, err=bkg_rms, mask=mask, \
    		minarea=minarea, filter_type=filter_type, \
    		deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, \
    		clean=clean, clean_param=clean_param, segmentation_map=True)

    if(args.flag_mode==2):
    	objects, seg_map = sep.extract(data_sub, thresh, err=data_err, mask=mask, \
    		minarea=minarea, filter_type=filter_type, \
    		deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, \
    		clean=clean, clean_param=clean_param, segmentation_map=True)

    t_end = time.time()
    print("Extraction time elapsed = ",t_end-t_start)

    print("Objects found = ", len(objects))

    #compute kron radius
    x = objects['x']
    y = objects['y']
    a = objects['a']
    b = objects['b']
    theta = objects['theta']
    theta = np.clip(theta,-0.5*np.pi,0.5*np.pi)

    #perform kron phot
    kronrad, krflag = sep.kron_radius(data_sub, x, y, a, b, theta, 6., mask=mask)
    kronrad[np.isnan(kronrad)]=int_nan

    #save the segmentation map
    seg_map[idx_nan] = -1
    seg_map[idx_mask] = -1

    fname_seg = args.name+".seg.fits"
    fseg = seg_map.astype(float)
    fseg[idx_nan] = np.nan
    fseg[idx_mask] = np.nan
    print("Writing segmentation map -- ",fname_seg)
    fits.writeto(fdir_outfiles + fname_seg, fseg, header_data, overwrite=True)

    #output the catalog
    print("Outputting the catalog")

    #find the ra and dec of sources
    x = objects['x']
    y = objects['y']
    pos = np.zeros((len(x),2))
    pos[:,0] = x
    pos[:,1] = y
    pos_sky = wcs_obj.wcs_pix2world(pos,0) 

    #write catalog of detections
    catstr += str(args.window_size)
    fout = fdir_outfiles + args.name + catstr + ".det.cat"
    write_det_catalog(fout,objects,pos_sky, kronrad, krflag, args.flag_sort)

    all_end = time.time()
    print("Total time elapsed = ",all_end-all_start)
    print("Done!")

if __name__ == "__main__": main()   

