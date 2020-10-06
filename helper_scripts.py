from astropy.io import fits
from astropy import wcs
import numpy as np

def write_det_catalog(fname,objects,pos_sky,kronrad,flag,flag_sort):
    fp = open(fname,"w")

    x = objects['x']
    y = objects['y']
    a = objects['a']
    b = objects['b']
    theta = objects['theta']
    npix = objects['npix']

    z = -9999
    s = "%6s " % "id"
    s += "%10s %10s " % ("x","y")
    s += "%9s  %9s " % ("ra","dec")
    s += "%7s %7s %7s " % ("a","b","theta")
    s += "%6s " % "npix"
    s += "%8s " % "r_Kron"
    s += "%8s " % ("Kron_flg")
    s += "\n"

    fp.write(s)

    if(flag_sort):
    	idx_rad = np.argsort(-1*kronrad)

    for j in range(len(objects)):
        if(flag_sort):
         	i = idx_rad[j]
        else:
        	i = j
        idout = i+1 
        s = "%6d %10.5f %10.5f %8.6f %8.6f " % (idout,x[i],y[i],pos_sky[i,0],pos_sky[i,1])
        s += "%7.4f %7.4f % 6.4f %6d %8.6f %8d " % (a[i],b[i],theta[i],npix[i], kronrad[i], flag[i])
        s += "\n"
        fp.write(s)
    fp.close()
    return 


def write_phot_catalog(fname,band,objects,pos_sky,kronrad,flux,fluxerr,flag,cflux,cfluxerr,cflag,flag_sort,flag_ids):
    
    fp = open(fname,"w")

    x = objects['x']
    y = objects['y']
    a = objects['a']
    b = objects['b']
    theta = objects['theta']
    npix = objects['npix']
    if(flag_ids):
      ido = objects['ido']

    z = -9999
    s = "%6s " % "id"
    s += "%10s %10s " % ("x","y")
    s += "%9s  %9s " % ("ra","dec")
    s += "%7s %7s %7s " % ("a","b","theta")
    s += "%6s " % "npix"
    s += "%8s " % "r_Kron"
    s += "%13s   " % ("NRC_"+band)
    s += "%13s " % ("NRC_"+band+"_err")
    s += "%8s " % ("AUTO_flg")
    s += "%12s   " % ("f70_"+band)
    s += "%12s " % ("f70_"+band+"_err")
    s += "%7s " % ("f70_flg")
    s += "\n"


    if(flag_sort):
    	idx_flux = np.argsort(-1*flux)
    #\tx\ty\tra\tdec\ta\tb\ttheta\tnpix\tKron rad [pix]\tNRC_%s\tNRC_%s_err\tAUTO_FLAG\tf_r70\tf_r70_err\tr70_FLAG\n" % (band,band)

    #s = "id\tx\ty\tra\tdec\ta\tb\ttheta\tnpix\tKron rad [pix]\tNRC_%s\tNRC_%s_err\tAUTO_FLAG\tf_r70\tf_r70_err\tr70_FLAG\n" % (band,band)
    fp.write(s)
    for j in range(len(objects)):
        #s = "%6d %10.5f %10.5f %8.6f %8.6f %7.4f %7.4f % 6.4f %6d %6.4f % 8.6f %8.6f %d %8.6f %8.6f %d\n" \
        #    % ,,kronrad[i], \
        #    flux[i],fluxerr[i],flag[i],cflux[i],cfluxerr[i],cflag[i])
        if(flag_sort):
         	i = idx_flux[j]
        else:
        	i = j
        if(flag_ids):
           idout = ido[i] 
        else:
           idout = i+1 
        s = "%6d %10.5f %10.5f %8.6f %8.6f " % (idout,x[i],y[i],pos_sky[i,0],pos_sky[i,1])
        s += "%7.4f %7.4f % 6.4f %6d " % (a[i],b[i],theta[i],npix[i])
        s += "%8.6f " %(kronrad[i])
        s += "%13.6f   %13.6f %8d" %(flux[i],fluxerr[i],flag[i])
        s += "%13.6f   %13.6f %7d" % (cflux[i],cfluxerr[i],cflag[i])
        s += "\n"
        fp.write(s)
    fp.close()

    return 

def read_fits(fname):
    hdul_data = fits.open(fname)
    data = hdul_data[0].data
    header_data = hdul_data[0].header
    data = data.byteswap().newbyteorder()
    return data, header_data


def make_objects(n):
    objects = np.zeros(n, dtype=[('npix','i8'),('x','f8'),('y','f8'),('a','f8'),('b','f8'),('theta','f8'),('ido','i8'), ('kronrad','f8'),('krflag','i8')])
    return objects