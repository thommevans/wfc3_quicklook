from __future__ import print_function
import math
import numpy as np
import scipy, scipy.ndimage, scipy.stats
import matplotlib.pyplot as plt
import pyfits
import os, pdb, time
import glob
import pickle
import batman, george, emcee, corner

"""
Script for generating a quicklook transmission/emission spectrum
from an HST/WFC3 dataset, starting at the raw *ima*fits frames.
"""

HOMEDIR = os.path.expanduser( '~' )

BANDPASS_FPATH_G141 = os.path.join( os.getcwd(), 'HST_throughputs/WFC3.IR.G141.1st.sens.2.fits' )
BANDPASS_FPATH_G102 = os.path.join( os.getcwd(), 'HST_throughputs/WFC3.IR.G102.1st.sens.2.fits' )

HST_ORB_PERIOD_DAYS = 96/60./24. # i.e. 96 minutes

ATLAS_FPATH = os.path.join( os.getcwd(), 'Kurucz/ip03k2new.pck' )
ATLAS_NEWGRID = True
ATLAS_TEFF = 4750.
ATLAS_LOGG = 4.5

# SI values for standard units:
AU = 1.496e11
RSUN = 6.955e8
RJUP = 7.149e7

#################################################################################
# Wrapper for running pipeline.

def main( switches={ 'extract_spectra':True, 'create_whitelc':True, 'fit_whitelc':True, \
                     'create_speclcs':True, 'fit_speclcs':True }, \
          white_mcmc={ 'ngroups':3, 'nwalkers':150, 'nburn1':100, \
                       'nburn2':500, 'nsteps':500 }, \
          save_rdiff_pngs=False ):
    """
    Run this routine from the command line. Provide as input a switches 
    dictionary to specify which pipeline steps are to be run.
    """

    t1 = time.time()

    # Get values for various parameters for the current dataset
    # in order to determine various pipeline options:
    red, wlc, slc, syspars = dataset_parameters()

    # Extract the spectra from the IMA files:
    if switches['extract_spectra']==True:
        spectra_fpaths = extract_spectra( red, save_rdiff_pngs=save_rdiff_pngs )
    else:
        spectra_fpaths = get_spectra_fpath( red )
    spectra_rlast_fpath = spectra_fpaths[0]    
    spectra_rlast_zapped_fpath = spectra_fpaths[1]    
    spectra_rdiff_fpath = spectra_fpaths[2] 
    spectra_rdiff_zapped_fpath = spectra_fpaths[3]

    # Create and fit the white lightcurve:
    if switches['create_whitelc']==True:
        whitelc_fpath = create_whitelc( wlc, spectra_rdiff_zapped_fpath, red )
    else:
        whitelc_fpath = get_whitelc_fpath( spectra_rdiff_zapped_fpath )
    if switches['fit_whitelc']==True:        
        white_fpaths = fit_whitelc( whitelc_fpath, syspars, red, \
                                    ngroups=white_mcmc['ngroups'], \
                                    nwalkers=white_mcmc['nwalkers'], \
                                    nburn1=white_mcmc['nburn1'], \
                                    nburn2=white_mcmc['nburn2'], \
                                    nsteps=white_mcmc['nsteps'] )
    else:
        white_fpaths = get_whitefit_fpath( whitelc_fpath )
    whitetxt_fpath, whitemcmc_fpath, whitemle_fpath = white_fpaths

    # Create the spectroscopic lightcurves:
    if switches['create_speclcs']==True:
        speclcs_fpath = create_speclcs( slc, spectra_rdiff_zapped_fpath, whitemle_fpath, red )
    else:
        speclcs_fpath = get_speclcs_fpath( spectra_rdiff_zapped_fpath )

    # Fit the spectroscopic lightcurves:
    if switches['fit_speclcs']==True:
        wavc, RpRs_vals, RpRs_uncs = fit_speclcs_ttrend_quick( whitemle_fpath, speclcs_fpath )

    t2 = time.time()
    print( '\n\nTotal time taken = {0:.2f} minutes\n\n'.format( ( t2-t1 )/60. ) )

    return None

#################################################################################
# Dataset parameters. 
# These should be edited as appropriate for the dataset under consideration.

def dataset_parameters():
    """
    Specify the parameters to be adopted for:
      - transiting planet system
      - data reduction
      - creating the white lightcurve
      - creating the spectroscopic lightcurves
    """

    # Parameters controlling the data reduction:
    red = {}
    red['config'] = 'G102' # or 'G141'
    red['ddir'] = os.path.join( HOMEDIR, 'extension/to/data/folder' )
    red['apradius'] = 120
    red['maskradius'] = 35
    red['ntrim_edge'] = 10
    red['smoothing_fwhm'] = None
    red['crossdisp_bound_ixs'] = [ 9, 257 ]
    red['trim_disp_ixs'] = [ 25, 257 ]
    red['shiftstretch_disp_ixs'] = [ 60, 250 ]
    red['bg_crossdisp_ixs'] = [ 7, 17 ]
    red['bg_disp_ixs'] = [ 50, 220 ]
    red['discard_first_exposure'] = True

    # Parameters controlling the white lightcurve:
    wlc = {}
    wlc['cuton_micron'] = -1
    wlc['cutoff_micron'] = 10
    
    # Parameters controlling the spectroscopic lightcurves:
    slc = {}
    slc['cuton_micron'] = 0.8
    slc['npix_perbin'] = 5 
    slc ['nchannels'] = 26 

    # Parameters containing information about the transiting system:
    syspars = {}
    syspars['tr_type'] = 'primary' # 'primary' or 'secondary'
    Rstar = 0.9*RSUN
    syspars['RpRs'] = (1.1*RJUP)/Rstar # planet-to-star radius ratio
    syspars['T0'] = 2454678.1234 # transit mid-time
    syspars['P'] = 3.45 # orbital period
    syspars['aRs'] = (0.05*AU)/Rstar # normalised semimajor axis 
    syspars['incl'] = 89.5 # orbital inclination (degrees)
    syspars['ecc'] = 0.0 # orbital eccentricity 
    syspars['omega'] = 90. # longitude of periastron (degrees)
    syspars['Tmid'] = syspars['T0'] # transit/eclipse mid-time

    return red, wlc, slc, syspars

#################################################################################
# Model priors for white lightcurve model.
# These can in theory be edited as appropriate for the dataset under consideration,
# but hopefully this is not required.

def lnprior_primary_forward( pars ):
    """
    Log( prior ) function for a primary transit dataset taken in forward
    scanning mode. Hopefully this default prior will be sufficient, but
    it can be edited if necessary.
    """
    A, lniLx, lniLy, lniLz, c0, c1, RpRs, delT = pars
    ##################################
    # Gamma prior for A:
    alpha = 1
    beta = 1e2
    if A<=0:
        lnA = -np.inf
    else:
        lnA = np.log( scipy.stats.gamma.pdf( A, alpha, loc=0, scale=1/beta ) )
    ##################################
    # Uniform prior for lniL:
    low = -5
    upp = 5
    lnlniLx = np.log( scipy.stats.uniform.pdf( lniLx, loc=low, scale=upp-low ) )
    lnlniLy = np.log( scipy.stats.uniform.pdf( lniLy, loc=low, scale=upp-low ) )
    lnlniLz = np.log( scipy.stats.uniform.pdf( lniLz, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for c0:
    low = 0.98
    upp = 1.02
    lnc0 = np.log( scipy.stats.uniform.pdf( c0, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for c1:
    low = -1
    upp = 1
    lnc1 = np.log( scipy.stats.uniform.pdf( c1, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for RpRs:
    low = 1e-6
    upp = 0.8
    lnRpRs = np.log( scipy.stats.uniform.pdf( RpRs, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for delT
    low = -1./24.
    upp = 1./24.
    lndelT = np.log( scipy.stats.uniform.pdf( delT, loc=low, scale=upp-low ) )
    return lnA + lnlniLx + lnlniLy + lnlniLz + lnc0 + lnc1 + lnRpRs + lndelT

def lnprior_primary_bidirection( pars ):
    """
    Log( prior ) function for a primary transit dataset taken in bidirection
    scanning mode. Hopefully this default prior will be sufficient, but
    it can be edited if necessary.
    """
    nscandir = 2
    # Systematics parameters for 1st scandir:
    A_f = pars[0]
    lniLx_f = pars[1]
    lniLy_f = pars[2]
    lniLz_f = pars[3]
    c0_f = pars[4]
    c1_f = pars[5]
    # Systematics parameters for 2nd scandir:
    A_r = pars[6]
    lniLx_r = pars[7]
    lniLy_r = pars[8]
    lniLz_r = pars[9]
    c0_r = pars[10]
    c1_r = pars[11]
    # Combine pairs of parameters:
    A = [ A_f, A_r ]
    lniLx = [ lniLx_f, lniLx_r ]
    lniLy = [ lniLy_f, lniLy_r ]
    lniLz = [ lniLz_f, lniLz_r ]
    c0 = [ c0_f, c0_r ]
    c1 = [ c1_f, c1_r ]
    # Systematics parameters for shared planet parameters:
    RpRs = pars[12]
    delT = pars[13]
    # Compute prior values for systematics parameters:
    lnA = lnlniLx = lnlniLy = lnlniLz = lnc0 = lnc1 = 0
    for i in range( nscandir ):
        ##################################
        # Gamma prior for A:
        alpha = 1
        beta = 1e2
        if A[i]<=0:
            lnA += -np.inf
        else:
            lnA += np.log( scipy.stats.gamma.pdf( A[i], alpha, loc=0, scale=1/beta ) )
        ##################################
        # Uniform prior for lniL:
        low = -5
        upp = 5
        lnlniLx += np.log( scipy.stats.uniform.pdf( lniLx[i], loc=low, scale=upp-low ) )
        lnlniLy += np.log( scipy.stats.uniform.pdf( lniLy[i], loc=low, scale=upp-low ) )
        lnlniLz += np.log( scipy.stats.uniform.pdf( lniLz[i], loc=low, scale=upp-low ) )
        ##################################
        # Uniform prior for c0:
        low = 0.98
        upp = 1.02
        lnc0 += np.log( scipy.stats.uniform.pdf( c0[i], loc=low, scale=upp-low ) )
        ##################################
        # Uniform prior for c1:
        low = -1
        upp = 1
        lnc1 += np.log( scipy.stats.uniform.pdf( c1[i], loc=low, scale=upp-low ) )
        ##################################
    # Compute prior values for shared planet parameters:
    ##################################
    # Uniform prior for RpRs:
    low = 1e-6
    upp = 0.8
    lnRpRs = np.log( scipy.stats.uniform.pdf( RpRs, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for delT
    low = -1./24.
    upp = 1./24.
    lndelT = np.log( scipy.stats.uniform.pdf( delT, loc=low, scale=upp-low ) )
    return lnA + lnlniLx + lnlniLy + lnlniLz + lnc0 + lnc1 + lnRpRs + lndelT

def lnprior_secondary_forward( pars ):
    """
    Log( prior ) function for a secondary eclipse dataset taken in forward
    scanning mode. Hopefully this default prior will be sufficient, but
    it can be edited if necessary.
    """
    A, lniLx, lniLy, lniLz, c0, c1, SecDepth, delT = pars
    ##################################
    # Gamma prior for A:
    alpha = 1
    beta = 1e2
    if A<=0:
        lnA = -np.inf
    else:
        lnA = np.log( scipy.stats.gamma.pdf( A, alpha, loc=0, scale=1/beta ) )
    ##################################
    # Uniform prior for lniL:
    low = -5
    upp = 5
    lnlniLx = np.log( scipy.stats.uniform.pdf( lniLx, loc=low, scale=upp-low ) )
    lnlniLy = np.log( scipy.stats.uniform.pdf( lniLy, loc=low, scale=upp-low ) )
    lnlniLz = np.log( scipy.stats.uniform.pdf( lniLz, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for c0:
    low = 0.98
    upp = 1.02
    lnc0 = np.log( scipy.stats.uniform.pdf( c0, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for c1:
    low = -1
    upp = 1
    lnc1 = np.log( scipy.stats.uniform.pdf( c1, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for SecDepth:
    low = -1e-2
    upp = 1e-2
    lnSecDepth = np.log( scipy.stats.uniform.pdf( SecDepth, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for delT
    low = -1./24.
    upp = 1./24.
    lndelT = np.log( scipy.stats.uniform.pdf( delT, loc=low, scale=upp-low ) )
    return lnA + lnlniLx + lnlniLy + lnlniLz + lnc0 + lnc1 + lnSecDepth + lndelT

  
def lnprior_secondary_bidirection( pars ):
    """
    Log( prior ) function for a secondary eclipse dataset taken in bidirection
    scanning mode. Hopefully this default prior will be sufficient, but
    it can be edited if necessary.
    """
    nscandir = 2
    # Systematics parameters for 1st scandir:
    A_f = pars[0]
    lniLx_f = pars[1]
    lniLy_f = pars[2]
    lniLz_f = pars[3]
    c0_f = pars[4]
    c1_f = pars[5]
    # Systematics parameters for 2nd scandir:
    A_r = pars[6]
    lniLx_r = pars[7]
    lniLy_r = pars[8]
    lniLz_r = pars[9]
    c0_r = pars[10]
    c1_r = pars[11]
    # Combine pairs of parameters:
    A = [ A_f, A_r ]
    lniLx = [ lniLx_f, lniLx_r ]
    lniLy = [ lniLy_f, lniLy_r ]
    lniLz = [ lniLz_f, lniLz_r ]
    c0 = [ c0_f, c0_r ]
    c1 = [ c1_f, c1_r ]
    # Systematics parameters for shared planet parameters:
    SecDepth = pars[12]
    delT = pars[13]
    # Compute prior values for systematics parameters:
    lnA = lnlniLx = lnlniLy = lnlniLz = lnc0 = lnc1 = 0
    for i in range( nscandir ):
        ##################################
        # Gamma prior for A:
        alpha = 1
        beta = 1e2
        if A[i]<=0:
            lnA += -np.inf
        else:
            lnA += np.log( scipy.stats.gamma.pdf( A[i], alpha, loc=0, scale=1/beta ) )
        ##################################
        # Uniform prior for lniL:
        low = -5
        upp = 5
        lnlniLx += np.log( scipy.stats.uniform.pdf( lniLx[i], loc=low, scale=upp-low ) )
        lnlniLy += np.log( scipy.stats.uniform.pdf( lniLy[i], loc=low, scale=upp-low ) )
        lnlniLz += np.log( scipy.stats.uniform.pdf( lniLz[i], loc=low, scale=upp-low ) )
        ##################################
        # Uniform prior for c0:
        low = 0.98
        upp = 1.02
        lnc0 += np.log( scipy.stats.uniform.pdf( c0[i], loc=low, scale=upp-low ) )
        ##################################
        # Uniform prior for c1:
        low = -1
        upp = 1
        lnc1 += np.log( scipy.stats.uniform.pdf( c1[i], loc=low, scale=upp-low ) )
        ##################################
    # Compute prior values for shared planet parameters:
    ##################################
    # Uniform prior for SecDepth:
    low = -1e-2
    upp = 1e-2
    lnSecDepth = np.log( scipy.stats.uniform.pdf( SecDepth, loc=low, scale=upp-low ) )
    ##################################
    # Uniform prior for delT
    low = -1./24.
    upp = 1./24.
    lndelT = np.log( scipy.stats.uniform.pdf( delT, loc=low, scale=upp-low ) )
    return lnA + lnlniLx + lnlniLy + lnlniLz + lnc0 + lnc1 + lnSecDepth + lndelT


#################################################################################
# Top-level reduction routines.

def extract_spectra( red, save_rdiff_pngs=False ):
    """
    Extracts the timeseries spectra from the raw *ima*fits data frames.
    """
    z = prep_frames( red, save_rdiff_pngs=save_rdiff_pngs )
    jd = z[0]
    scandirs = z[1]
    bg_ppix = z[2]
    ecounts2d_rlast = z[3]
    ecounts2d_rlast_zapped = z[4]
    ecounts2d_rdiff = z[5]
    ecounts2d_rdiff_zapped = z[6]
    # Calculate the HST orbital phase values:
    delt = jd - jd[0]
    hstphase = np.mod( delt, HST_ORB_PERIOD_DAYS )/float( HST_ORB_PERIOD_DAYS )
    ixs = hstphase>0.5
    hstphase[ixs] -= 1
    # Split the orbits:
    orbixs = split_orbixs( delt*24 )
    torb = np.zeros( jd.size )
    for i in orbixs:
        torb[i] = jd[i]-jd[i][0]
    auxvars = {}
    auxvars['jd'] = jd
    auxvars['torb'] = torb
    auxvars['hstphase'] = hstphase
    auxvars['background_ppix'] = bg_ppix
    f2d = [ ecounts2d_rlast, ecounts2d_rlast_zapped, ecounts2d_rdiff, ecounts2d_rdiff_zapped ]
    suffix = [ '.pkl', '.zapped.pkl' ]
    opaths = get_spectra_fpath( red )
    output = []
    for k in range( 4 ):
        spec = extract_spatscan_spectra( f2d[k], ap_radius=red['apradius'], cross_axis=0, \
                                         disp_axis=1, frame_axis=2 )
        wavsol = get_wavsol( red['config'], spec['ecounts1d'][0,:], make_plot=False )
        pix2micron = np.median( np.diff( wavsol ) )
        ss = calc_spectra_variations( spec['ecounts1d'], spec['ecounts1d'][0,:], \
                                      max_wavshift_pixel=2, dwav=0.001, \
                                      smoothing_fwhm=red['smoothing_fwhm'], \
                                      disp_bound_ixs=red['shiftstretch_disp_ixs'] )
        auxvars['x'] = np.ones( spec['cdcs'].size )
        auxvars['cdcs'] = spec['cdcs']
        auxvars['wavshift_pixels'] = ss[1]
        auxvars['wavshift_micron'] = ss[1]*pix2micron
        outputk = { 'auxvars':auxvars, 'scandirs':scandirs, \
                    'wavsol_micron':wavsol, 'ecounts':spec['ecounts1d'] }
        ofile = open( opaths[k], 'wb' )
        pickle.dump( outputk, ofile )
        ofile.close()
        output += [ outputk ]
    fig_opath = plot_basic_timeseries( output, red )
    print( '\nSaved:' )
    for k in opaths+[fig_opath]:
        print( '\n{0}\n'.format( k ) )
    return opaths

def create_whitelc( z, spectra_fpath, red ):
    """
    Creates the white lightcurve.
    """
    ifile = open( spectra_fpath, 'rb' )
    spectra = pickle.load( ifile )
    ifile.close()
    wavsol_micron = spectra['wavsol_micron']
    ecounts1d = spectra['ecounts']
    ndisp = len( wavsol_micron )
    x = np.arange( ndisp )
    ixs = ( wavsol_micron>=z['cuton_micron'] )*( wavsol_micron<=z['cutoff_micron'] )
    dispixs = [ x[ixs][0], x[ixs][-1] ]
    whitelc = make_lc( spectra, red, dispixs=dispixs )
    opath = get_whitelc_fpath( spectra_fpath )
    ofile = open( opath, 'wb' )
    pickle.dump( whitelc, ofile )
    ofile.close()
    scanmode, ixsf, ixsr = determine_scanmode( whitelc['scandirs'] )
    plt.ioff()
    fig = plt.figure( figsize=[12,12] )
    ax1 = fig.add_subplot( 4, 1, 1 )
    ax2 = fig.add_subplot( 4, 1, 2, sharex=ax1 )
    ax3 = fig.add_subplot( 4, 1, 3, sharex=ax1 )
    ax4 = fig.add_subplot( 4, 1, 4, sharex=ax1 )
    if ( scanmode=='forward' )+( scanmode=='reverse' ):
        c = 'k'
        ax1.plot( whitelc['jd'], whitelc['flux'], 'o', mfc=c, mec=c )
        ax2.plot( whitelc['jd'], whitelc['auxvars']['background_ppix'], 'o', mfc=c, mec=c )
        ax3.plot( whitelc['jd'], whitelc['auxvars']['wavshift_pixels'], 'o', mfc=c, mec=c )
        ax4.plot( whitelc['jd'], whitelc['auxvars']['cdcs'], 'o', mfc=c, mec=c )
    elif scanmode=='bidirection':
        ixs = [ ixsf, ixsr ]
        c = [ 'c', 'r' ]
        for i in range( 2 ):
            ax1.plot( whitelc['jd'][ixs[i]], whitelc['flux'][ixs[i]], 'o', mfc=c[i], mec=c[i] )
            ax2.plot( whitelc['jd'][ixs[i]], whitelc['auxvars']['background_ppix'][ixs[i]], \
                      'o', mfc=c[i], mec=c[i] )
            ax3.plot( whitelc['jd'][ixs[i]], whitelc['auxvars']['wavshift_pixels'][ixs[i]], \
                      'o', mfc=c[i], mec=c[i] )
            ax4.plot( whitelc['jd'][ixs[i]], whitelc['auxvars']['cdcs'][ixs[i]], 'o', mfc=c[i], mec=c[i] )
    ax1.set_ylabel( 'flux' )
    ax2.set_ylabel( 'background' )
    ax3.set_ylabel( 'dx (pix)' )
    ax4.set_ylabel( 'dy (pix)' )
    ax4.set_xlabel( 'JD' )
    fig.savefig( opath.replace( '.pkl', '.pdf' ) )
    print( '\nSaved:\n{0}\n'.format( opath ) )
    return opath

def create_speclcs( z, spectra_fpath, whitefit_fpath, red ):
    """
    Creates the spectroscopic lightcurves.
    """
    opath = get_speclcs_fpath( spectra_fpath )
    ifile = open( spectra_fpath, 'rb' )
    spectra = pickle.load( ifile )
    ifile.close()
    ifile = open( whitefit_fpath, 'rb' )
    whitefit = pickle.load( ifile )
    ifile.close()
    chixs = get_spec_chixs( z, spectra )
    basic = prep_speclcs_basic( spectra, spectra_fpath, whitefit, chixs, red )
    shiftstretch = prep_speclcs_shiftstretch( spectra, spectra_fpath, whitefit, chixs, red )
    output = {}
    output['jd'] = basic['jd']
    output['scanmode'] = basic['scanmode']
    output['scandirs'] = basic['scandirs']
    output['flux_raw'] = basic['flux_raw']
    output['uncs_raw'] = basic['uncs_raw']
    output['flux_cm1'] = basic['flux_cm1']
    output['uncs_cm1'] = basic['uncs_cm1']
    output['flux_cm2'] = basic['flux_cm2']
    output['uncs_cm2'] = basic['uncs_cm2']
    output['flux_ss'] = shiftstretch['flux_ss']
    output['uncs_ss'] = shiftstretch['uncs_ss']
    output['wavedges'] = basic['wavedges']
    output['ld_quad'] = basic['ld_quad']
    output['ld_nonlin'] = basic['ld_nonlin']
    output['auxvars'] = basic['auxvars']
    output['nchan'] = basic['nchan']    
    ofile = open( opath, 'wb' )
    pickle.dump( output, ofile )
    ofile.close()
    print( '\nSaved:\n{0}\n'.format( opath ) )
    return opath

#################################################################################
# Lower-level reduction routines.

def prep_frames( red, save_rdiff_pngs=False ):
    """
    Prepare the read-differenced frames.
    """

    if ( red['smoothing_fwhm']==None )+( red['smoothing_fwhm']==0 ):
        red['smoothing_str'] = 'unsmoothed'
        red['smoothing_fwhm'] = 0.0
    else:
        red['smoothing_str'] = 'smooth{0:.2f}pix'.format( red['smoothing_fwhm'] )
    ecounts2d_rlast, ecounts2d_rdiff, tstarts, exptimes, scandirs, bg_ppix, fs = get_frames( red )

    # Compute the mid-exposure times:
    mjd = tstarts + 0.5*exptimes/60./60./24.
    jd = mjd + 2400000.5

    # Arrange in chronological order and remove the 
    # initial acquisition exposure:
    ixs = np.argsort( jd )
    fs = fs[ixs] # list of chronologically-sorted filenames will be saved below
    mjd = mjd[ixs]
    jd = jd[ixs]
    exptimes = exptimes[ixs]
    tstarts = tstarts[ixs]
    scandirs = scandirs[ixs]
    bg_ppix = bg_ppix[ixs]
    ecounts2d_rlast = ecounts2d_rlast[:,:,ixs]
    ecounts2d_rdiff = ecounts2d_rdiff[:,:,ixs]

    zap_rlast = clean_cosmic_rays( ecounts2d_rlast, inspect_each_frame=False )
    ecounts2d_rlast_zapped = zap_rlast[0]
    zap_rdiff = clean_cosmic_rays( ecounts2d_rdiff, inspect_each_frame=False )
    ecounts2d_rdiff_zapped = zap_rdiff[0]
    save_reconstructed_frames( jd, ecounts2d_rdiff, red, zap_rdiff, fs, save_rdiff_pngs=save_rdiff_pngs )
    return jd, scandirs, bg_ppix, ecounts2d_rlast, ecounts2d_rlast_zapped, ecounts2d_rdiff, ecounts2d_rdiff_zapped


def clean_cosmic_rays( ecounts2d, nsig_cull_transient=8, nsig_cull_static=10, niter=1, inspect_each_frame=False ):
    """
    Routine for identifying static and transient bad pixels in a spectroscopic data cube. 

    Inputs:
    ecounts2d - NxMxK data cube where N is cross-dispersion, M is dispersion, K is frame number.
    nsig_cull_transient - threshold for flagging transient bad pixels.
    nsig_cull_static - threshold for flagging static bad pixels.
    niter - number of iterations to be used

    Outputs:
    ecounts2d_zapped - NxMxK cube containing the data with bad pixels corrected.
    transient_bad_pixs - NxMxK cube containing 1's for transient bad pixels and 0's otherwise
    static_bad_pixs - NxMxK cube containing 1's for static bad pixels and 0's otherwise
    ecounts2d_medfilt - NxMxK cube containing nominal PSF for each frame made using median filter
    """
    print( '\n\nCleaning cosmic rays:' )
    # Initialise arrays to hold all the outputs:
    ndisp, ncross, nframes = np.shape( ecounts2d )
    ecounts2d_zapped = np.zeros( [ ndisp, ncross, nframes ] ) # array for corrected data frames
    ecounts2d_medfilt = np.zeros( [ ndisp, ncross, nframes ] ) # array for median-filter frames
    transient_bad_pixs = np.zeros( [ ndisp, ncross, nframes ] ) # array for transient bad pixels
    static_bad_pixs = np.zeros( [ ndisp, ncross, nframes ] ) # array for static bad pixels
    # First apply a Gaussian filter to the pixel values
    # along the time axis of the data cube:
    ecounts2d_smoothed = scipy.ndimage.filters.gaussian_filter1d( ecounts2d, sigma=5, axis=2 )
    ecounts2d_smoothsub = ecounts2d - ecounts2d_smoothed # pixel deviations from smoothed time series
    med2d = np.median( ecounts2d_smoothsub, axis=2 )  # median deviation for each pixel
    stdv2d = np.std( ecounts2d_smoothsub, axis=2 ) # standard deviation in the deviations for each pixel
    # Loop over the data frames:
    for i in range( nframes ):
        ecounts2d_zapped[:,:,i] = ecounts2d[:,:,i]
        # Identify and replace transient bad pixels, possibly iterating more than once:
        for k in range( niter ):
            # Find the deviations of each pixel in the current frame in terms of 
            # number-of-sigma relative to the corresponding smoothed time series for
            # each pixel:
            ecounts2d_smoothsub = ecounts2d_zapped[:,:,i] - ecounts2d_smoothed[:,:,i]
            dsig_transient = np.abs( ( ecounts2d_smoothsub-med2d )/stdv2d )
            # Flag the outliers:
            ixs_transient = ( dsig_transient>nsig_cull_transient )
            # Create a median-filter frame by taking the median of 5 pixels along the
            # cross-dispersion axis for each pixel, to be used as a nominal PSF:
            medfilt_ik = scipy.ndimage.filters.median_filter( ecounts2d_zapped[:,:,i], size=[5,1] )
            # Interpolate any flagged pixels:
            ecounts2d_zapped[:,:,i][ixs_transient] = medfilt_ik[ixs_transient]
            # Record the pixels that were flagged in the transient bad pixel map:
            transient_bad_pixs[:,:,i][ixs_transient] = 1
        ntransient = transient_bad_pixs[:,:,i].sum() # number of transient bad pixels for current frame
        # Identify and replace static bad pixels, possibly iterating more than once:
        for k in range( niter ):
            # Create a median-filter frame by taking the median of 5 pixels along the
            # cross-dispersion axis for each pixel, to be used as a nominal PSF:
            medfilt_ik = scipy.ndimage.filters.median_filter( ecounts2d_zapped[:,:,i], size=[5,1] )
            # Find the deviations of each pixel in the current frame in terms of 
            # number-of-sigma relative to the nominal PSF:
            dcounts_static = ecounts2d_zapped[:,:,i] - medfilt_ik
            stdv_static = np.std( dcounts_static )
            dsig_static = np.abs( dcounts_static/stdv_static )
            # Flag the outliers:
            ixs_static = ( dsig_static>nsig_cull_static )
            # Interpolate any flagged pixels:
            ecounts2d_zapped[:,:,i][ixs_static] = medfilt_ik[ixs_static]
            # Record the pixels that were flagged in the static bad pixel map:
            static_bad_pixs[:,:,i][ixs_static] = 1
        nstatic = static_bad_pixs[:,:,i].sum() # number of transient bad pixels for current frame
        ecounts2d_medfilt[:,:,i] = medfilt_ik # record the nominal PSF for the current frame
        print( '... frame {0} of {1}: ntransient={2}, nstatic={3}'.format( i+1, nframes, ntransient, nstatic ) )
        if inspect_each_frame==True:
            plt.close( 'all' )
            fig = plt.figure( figsize=[24,8] )
            ax1 = fig.add_subplot( 151 )
            cax1 = ax1.imshow( transient_bad_pixs[:,:,i], interpolation='nearest', \
                               aspect='auto', cmap=matplotlib.cm.Reds )
            cbar1 = fig.colorbar( cax1 )
            ax1.set_title( 'N-sigma transient - {0} flagged'.format( ntransient ) )
            ax2 = fig.add_subplot( 152, sharex=ax1, sharey=ax1 )
            cax2 = ax2.imshow( static_bad_pixs[:,:,i], interpolation='nearest', \
                               aspect='auto', cmap=matplotlib.cm.Reds )
            cbar2 = fig.colorbar( cax2 )
            ax2.set_title( 'N-sigma static - {0} flagged'.format( nstatic ) )
            vmin = ecounts2d[:,:,i].min()
            vmax = ecounts2d[:,:,i].max()
            ax3 = fig.add_subplot( 153, sharex=ax1, sharey=ax1 )
            cax3 = ax3.imshow( ecounts2d[:,:,i], interpolation='nearest', \
                               aspect='auto', cmap=matplotlib.cm.Reds, vmin=vmin, vmax=vmax )
            cbar3 = fig.colorbar( cax3 )
            ax3.set_title( 'Raw ecounts2d' )
            ax4 = fig.add_subplot( 154, sharex=ax1, sharey=ax1 )
            cax4 = ax4.imshow( medfilt_ik, interpolation='nearest', aspect='auto', \
                               cmap=matplotlib.cm.Reds, vmin=vmin, vmax=vmax )
            cbar4 = fig.colorbar( cax4 )
            ax4.set_title( 'Median-filter' )
            ax5 = fig.add_subplot( 155, sharex=ax1, sharey=ax1 )
            cax5 = ax5.imshow( ecounts2d_zapped[:,:,i], interpolation='nearest', \
                               aspect='auto', cmap=matplotlib.cm.Reds, vmin=vmin, vmax=vmax )
            cbar5 = fig.colorbar( cax5 )
            ax5.set_title( 'Zapped ecounts2d' )
            fig.suptitle( 'Frame {0} of {1}'.format( i+1, nframes ) )
            pdb.set_trace()

    return ecounts2d_zapped, transient_bad_pixs, static_bad_pixs, ecounts2d_medfilt

def save_reconstructed_frames( jd, ecounts2d, red, zap, fs, save_rdiff_pngs=False ):
    """
    Creates a data cube containing the reconstructed frames obtained by
    the read-differencing process.
    """
    ecounts2d_zapped = zap[0]
    cosmic_ray_pixs = zap[1]
    static_bad_pixs = zap[2]
    ecounts2d_medfilt = zap[3]
    odir = os.path.join( os.getcwd(), 'rdiff_frames' )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    nscan, ndisp, nframes = np.shape( ecounts2d )
    ofilename = '{0}.apradius{1:.2f}pix.{2}.maskrad{3:.0f}pix.frames.rdiff.pkl'\
                .format( red['config'], red['apradius'], red['smoothing_str'], red['maskradius'] )
    opath = os.path.join( odir, ofilename )
    ofile = open( opath, 'wb' )
    output = { 'jd':jd, 'ecounts2d':ecounts2d }
    pickle.dump( output, ofile )
    ofile.close()
    ofilename = '{0}.apradius{1:.2f}pix.{2}.maskrad{3:.0f}pix.frames.rdiff.zapped.pkl'\
                .format( red['config'], red['apradius'], red['smoothing_str'], red['maskradius'] )
    opath_zapped = os.path.join( odir, ofilename )
    ofile = open( opath_zapped, 'wb' )
    output = { 'jd':jd, 'ecounts2d':ecounts2d_zapped, \
               'cosmic_ray_pixs':cosmic_ray_pixs, 'static_bad_pixs':static_bad_pixs, \
               'ecounts2d_medfilt':ecounts2d_medfilt }
    pickle.dump( output, ofile )
    ofile.close()
    print( '\nSaved rdiff frames:\n{0}\n{1}'.format( opath, opath_zapped ) )

    if save_rdiff_pngs==True:
        print( '\nSaving figures of 2D reconstructed images:' )
        plt.ioff()
        for i in range( nframes ):
            d = ecounts2d[:,:,i]
            plt.figure( figsize=[10,10] )
            plt.imshow( d, interpolation='nearest', vmin=0, vmax=25000 )
            plt.colorbar()
            opath = fs[i].replace( '.fits', '.png' )
            plt.savefig( opath )
            plt.close()
            d = ecounts2d_zapped[:,:,i]
            plt.figure( figsize=[10,10] )
            plt.imshow( d, interpolation='nearest', vmin=0, vmax=25000 )
            plt.colorbar()
            opath_zapped = fs[i].replace( '.fits', '.zapped.png' )
            plt.savefig( opath_zapped )
            plt.close()
            print( '... {0} + {1}'.format( opath, opath_zapped ) )
        plt.ion()
    odir = os.path.join( os.getcwd(), 'rdiff_frames' )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    return None

def extract_spatscan_spectra( image_cube, ap_radius=60, ninterp=10000, cross_axis=0, disp_axis=1, frame_axis=2 ):
    """
    Given a data cube of 2D image frames, determines the spatial scan centres
    and extracts the spectra by integrating within a specified aperture.
    """    
    z = np.shape( image_cube )
    ncross = z[cross_axis]
    ndisp = z[disp_axis]
    nframes = z[frame_axis]
    spectra = np.zeros( [ nframes, ndisp ] )
    cdcs = np.zeros( nframes )
    x = np.arange( ncross )
    nf = int( ninterp*len( x ) )
    xf = np.r_[ x.min():x.max():1j*nf ]
    print( '\nExtracting spectra from 2D images:' )
    for i in range( nframes ):
        print( '... image {0} of {1}'.format( i+1, nframes ) )
        image = image_cube[:,:,i]
        # Extract the cross-dispersion profile, i.e. along
        # the axis of the spatial scan:
        cdp = np.sum( image, axis=disp_axis )
        # Interpolate cross-dispersion profile to finer grid
        # in order to track sub-pixel shifts:
        cdpf = np.interp( xf, x, cdp )
        # Only consider points above the background level, 
        # otherwise blank sky will bias the result:
        thresh = cdp.min() + 0.1*( cdp.max()-cdp.min() )
        ixs = ( cdpf>thresh )
        # Determine the center of the scan by taking the
        # point midway between the edges:
        cdcs[i] = np.mean( xf[ixs] )
        # Determine the cross-dispersion coordinates between
        # which the integration will be performed:
        xmin = max( [ 0, cdcs[i] - ap_radius ] )
        xmax = min( [ cdcs[i] + ap_radius, ncross ] )
        # Determine the rows that are fully contained
        # within the aperture and integrate along the
        # cross-dispersion axis:
        xmin_full = int( np.ceil( xmin ) )
        xmax_full = int( np.floor( xmax ) )
        ixs_full = ( x>=xmin_full )*( x<=xmax_full )
        spectra[i,:] = np.sum( image[ixs_full,:], axis=cross_axis )        
        # Determine any rows that are partially contained
        # within the aperture at either end of the scan and
        # add their weighted contributions to the spectrum:
        xlow_partial = xmin_full - xmin
        spectra[i,:] += xlow_partial*image[xmin_full-1,:]
        xupp_partial = xmax - xmax_full
        spectra[i,:] += xupp_partial*image[xmax_full+1,:]
    return { 'cdcs':cdcs, 'ecounts1d':spectra }

def calc_spectra_variations( spectra, ref_spectrum, max_wavshift_pixel=5, dwav=0.01, \
                             smoothing_fwhm=None, disp_bound_ixs=[] ):
    """
    Performs shift+stretch.
    Returns:
      dspec - nframes x ndisp array containing the rescaled shifted spectrum minus the 
          reference spectrum.
      wavshifts - The amounts by which the reference spectrum had to be shifted along the shift
          in pixels along the dispersion axis in pixels to match the individual spectra.
      vstretches - The amounts by which the reference spectrum have to be vertically stretched
          to give the best match to the individual spectra.
    """
    # Standard options for WFC3 chip axes:
    frame_axis = 0
    disp_axis = 1
    nframes, ndisp = np.shape( spectra )

    # Convert smoothing fwhm to the standard deviation of the
    # Gaussian kernel, and smooth the reference spectrum:
    if smoothing_fwhm==None:
        smoothing_fwhm = 0.0
    if ( smoothing_fwhm>0 ):
        smoothing_sig = smoothing_fwhm/2./np.sqrt( 2.*np.log( 2. ) )
        ref_spectrum = scipy.ndimage.filters.gaussian_filter1d( ref_spectrum, smoothing_sig )

    # Interpolate the reference spectrum on to a grid of
    # increments equal to the dwav shift increment:
    dwavs = np.r_[-max_wavshift_pixel:max_wavshift_pixel+dwav:dwav]
    nshifts = len( dwavs )
    pad = max_wavshift_pixel+1
    x = np.arange( ndisp )
    xi = np.arange( -pad, ndisp+pad )
    z = np.zeros( pad )
    ref_spectrumi = np.concatenate( [ z, ref_spectrum, z ] )
    interpf = scipy.interpolate.interp1d( xi, ref_spectrumi, kind='cubic' )
    shifted = np.zeros( [ nshifts, ndisp ] )
    for i in range( nshifts ):
        shifted[i,:] = interpf( x+dwavs[i] )

    # Now loop over the individual spectra and determine which
    # of the shifted reference spectra gives the best match:
    print( '\nDetermining shifts and stretches:' )
    wavshifts = np.zeros( nframes )
    vstretches = np.zeros( nframes )
    dspec = np.zeros( [ nframes, ndisp ] )
    enoise = np.zeros( [ nframes, ndisp ] )
    ix0 = disp_bound_ixs[0]
    ix1 = disp_bound_ixs[1]
    A = np.ones([ndisp,2])
    A = np.column_stack( [ A, np.arange( ndisp ) ] ) # testing
    coeffs = []
    for i in range( nframes ):
        print( i+1, nframes )
        rms_i = np.zeros( nshifts )
        diffs = np.zeros( [ nshifts, ndisp ] )
        vstretches_i = np.zeros( nshifts )
        for j in range( nshifts ):
            A[:,1] = shifted[j,:]
            b = np.reshape( spectra[i,:], [ ndisp, 1 ] )
            res = np.linalg.lstsq( A, b )
            c = res[0].flatten()
            fit = np.dot( A, c )
            vstretches_i[j] = c[1]
            diffs[j,:] = spectra[i,:] - fit
            rms_i[j] = np.sqrt( np.mean( diffs[j,:][ix0:ix1+1]**2. ) )
        ix = np.argmin( rms_i )
        dspec[i,:] = diffs[ix,:]
        enoise[i,:] = np.sqrt( spectra[i,:] )
        wavshifts[i] = dwavs[ix]
        vstretches[i] = vstretches_i[ix]
    return dspec, wavshifts, vstretches, enoise

def make_lc( spectra, red, dispixs=[], ld3d=False ):
    """
    Produces a lightcurve by summing timeseries spectra
    between specified dispersion columns.
    """
    ixl = dispixs[0]
    ixu = dispixs[1]
    cuton_micron = spectra['wavsol_micron'][ixl]
    cutoff_micron = spectra['wavsol_micron'][ixu]
    flux = np.sum( spectra['ecounts'][:,ixl:ixu+1], axis=1 )
    uncs = np.sqrt( flux )
    f0 = flux[-1]
    flux /= f0
    uncs /= f0
    print( '\nSumming flux between dispersion columns {0:.0f}-{1:.0f}\n'.format( ixl, ixu ) )
    ld_coeffs = get_ld_py( cuton_micron=cuton_micron, cutoff_micron=cutoff_micron, \
                           config=red['config'], stagger3d=False )
    ld_quad = ld_coeffs['quadratic']
    ld_nonlin = ld_coeffs['fourparam_nonlin']
    output = {}
    output['jd'] = spectra['auxvars']['jd']
    output['scandirs'] = spectra['scandirs']
    output['flux'] = flux
    output['uncs'] = uncs
    output['cuton_micron'] = cuton_micron
    output['cutoff_micron'] = cutoff_micron
    output['ld_quad'] = ld_quad
    output['ld_nonlin'] = ld_nonlin
    output['auxvars'] = spectra['auxvars']
    # Add standardised auxiliary variables:
    for key in output['auxvars'].copy().keys():
        newkey = '{0}v'.format( key )
        v = output['auxvars'][key]
        output['auxvars'][newkey] = ( v-np.mean( v ) )/np.std( v )
    return output

def prep_speclcs_shiftstretch( spectra, spectra_fpath, whitefit, chixs, red ):
    """
    Generates spectroscopic lightcurves using the shift+stretch method.
    """
    wavsol = spectra['wavsol_micron']
    cullixs = whitefit['cullixs']
    white_psignal = whitefit['mle_model']['psignal']
    if whitefit['scanmode']=='bidirection':
        ixs = np.argsort( np.concatenate( whitefit['mle_model']['jd'] ) )
        white_psignal = np.concatenate( white_psignal )[ixs]
    ecounts = spectra['ecounts'][cullixs,:]
    ixs_in = white_psignal<1
    ndat, ndisp = np.shape( ecounts )
    ixs_out = np.concatenate( [ np.arange( ixs_in[0]-10 ), np.arange( ixs_in[-1]+10, ndat ) ] )
    refspec = np.median( ecounts[ixs_out,:] ,axis=0 )
    ss = calc_spectra_variations( ecounts, refspec, max_wavshift_pixel=2, dwav=0.001, \
                                  smoothing_fwhm=red['smoothing_fwhm'], \
                                  disp_bound_ixs=red['shiftstretch_disp_ixs'] )
    dspec, wavshifts, vstretches, enoise = ss
    # Normalise the residuals and uncertainties:
    for i in range( ndat ):
        dspec[i,:] /= refspec
        enoise[i,:] /= refspec

    # Construct the spectroscopic lightcurves using data that is 
    # contained within the range defined by disp_bound_ixs:
    nch = len( chixs )
    wavedges = np.zeros( [ nch, 2 ] )
    speclcs_flux_ss = np.zeros( [ nch, ndat ] )
    speclcs_uncs_ss = np.zeros( [ nch, ndat ] )
    for i in range( nch ):        
        a = chixs[i][0]
        b = chixs[i][1]
        wavedges[i,0] = wavsol[a]
        wavedges[i,1] = wavsol[b]
        # Bin the differential fluxes over the current channel:
        dspec_binned = np.mean( dspec[:,a:b+1], axis=1 )
        # Since the differential fluxes correspond to the raw spectroscopic
        # fluxes corrected for wavelength-common-mode systematics minus the 
        # white transit, we simply add back in the white transit signal to
        # obtain the systematics-corrected spectroscopic lightcurve:
        speclcs_flux_ss[i,:] = dspec_binned + white_psignal

        # Computed the binned uncertainties for the wavelength channel:
        for j in range( ndat ):
            fsum = ecounts[j,a:b+1].sum()
            speclcs_uncs_ss[i,j] = np.sqrt( fsum )/float( fsum )
    return { 'flux_ss':speclcs_flux_ss, 'uncs_ss':speclcs_uncs_ss }

def prep_speclcs_basic( spectra, spectra_fpath, whitefit, chixs, red ):
    """
    Generates spectroscopic lightcurves using the basic method of summing
    spectra within dispersion column channels.
    """
    cullixs = whitefit['cullixs']
    config = whitefit['red']['config']
    syspars = whitefit['syspars']
    nch = len( chixs )
    wavedges = np.zeros( [ nch, 2 ] )
    speclcs_flux_raw = []
    speclcs_uncs_raw = []
    ld_quad = np.zeros( [ nch, 2 ] )
    ld_nonlin = np.zeros( [ nch, 4 ] )
    for i in range( nch ):
        speclc_i = make_lc( spectra, red, dispixs=chixs[i] )
        speclcs_flux_raw += [ speclc_i['flux'] ]
        speclcs_uncs_raw += [ speclc_i['uncs'] ]
        wavedges[i,0] = speclc_i['cuton_micron']
        wavedges[i,1] = speclc_i['cutoff_micron']
        ld_quad[i,:] = speclc_i['ld_quad']
        ld_nonlin[i,:] = speclc_i['ld_nonlin']
    #cullixs = whitefit['whitelc']['cullixs']
    speclcs_flux_raw = np.row_stack( speclcs_flux_raw )[:,cullixs]
    speclcs_uncs_raw = np.row_stack( speclcs_uncs_raw )[:,cullixs]
    jd = whitefit['whitelc']['jd']#[cullixs]
    whiteflux = whitefit['whitelc']['flux'][cullixs]
    ndat = len( jd[cullixs] )

    speclcs_flux_cm1 = np.zeros( [ nch, ndat ] )
    speclcs_uncs_cm1 = np.zeros( [ nch, ndat ] )
    speclcs_flux_cm2 = np.zeros( [ nch, ndat ] )
    speclcs_uncs_cm2 = np.zeros( [ nch, ndat ] )

    jd_base = whitefit['mle_model']['jd']
    psignal_base = whitefit['mle_model']['psignal']
    if whitefit['scanmode']=='bidirection':
        jd_base = np.concatenate( jd_base )
        ixs = np.argsort( jd_base )
        psignal_base = np.concatenate( psignal_base )[ixs]
    for i in range( nch ):
        syspars['ld'] = None
        speclcs_flux_cm1_i = speclcs_flux_raw[i,:]/whiteflux
        speclcs_flux_cm1_i *= psignal_base
        speclcs_flux_cm1[i,:] = speclcs_flux_cm1_i
        speclcs_uncs_cm1[i,:] = speclcs_uncs_raw[i,:]/whiteflux
        speclcs_flux_cm2_i = speclcs_flux_raw[i,:]/speclcs_flux_raw[i,-1]-whiteflux/whiteflux[-1]
        speclcs_flux_cm2_i += psignal_base
        speclcs_flux_cm2[i,:] = speclcs_flux_cm2_i
        speclcs_uncs_cm2[i,:] = speclcs_uncs_raw[i,:]/speclcs_flux_raw[-1,i]
    output = {}
    output['jd'] = jd[cullixs]
    output['scanmode'] = whitefit['scanmode']
    output['scandirs'] = whitefit['scandirs']
    output['flux_raw'] = speclcs_flux_raw
    output['uncs_raw'] = speclcs_uncs_raw
    output['flux_cm1'] = speclcs_flux_cm1
    output['uncs_cm1'] = speclcs_uncs_cm1
    output['flux_cm2'] = speclcs_flux_cm2
    output['uncs_cm2'] = speclcs_uncs_cm2
    output['wavedges'] = wavedges
    output['ld_quad'] = ld_quad
    output['ld_nonlin'] = ld_nonlin
    output['auxvars'] = {}
    for key in spectra['auxvars'].keys():
        output['auxvars'][key] = spectra['auxvars'][key][cullixs]
    output['nchan'] = nch
    return output


#################################################################################
# Standard fitting routines.

def fit_whitelc( whitelc_fpath, syspars, red, ngroups=3, nwalkers=150, nburn1=100, nburn2=500, nsteps=500 ):
    """
    Fits the white lightcurve using a GP systematics model (implemented with george) and
    affine-invariant MCMC (implemented with emcee).
    """
    opath_txt, opath_mcmc, opath_mle = get_whitefit_fpath( whitelc_fpath )
    odir = os.path.dirname( opath_txt )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )

    ifile = open( whitelc_fpath, 'rb' )
    whitelc = pickle.load( ifile )
    ifile.close()
    jd = whitelc['jd']
    scandirs = whitelc['scandirs']
    cullixs = get_cullixs( jd, red['discard_first_exposure'] )#whitelc['cullixs']
    keys = [ 'jdv', 'hstphasev', 'wavshift_pixelsv', 'cdcsv' ]
    ivars = {}
    for k in keys:
        ivars[k] = whitelc['auxvars'][k][cullixs]
    scandirs = scandirs[cullixs]
    scanmode, forward, reverse = determine_scanmode( scandirs )
    
    # White lightcurve dataset to be fit:
    jd = jd[cullixs]
    t = ( jd-np.mean( jd ) )/np.std( jd )
    flux = whitelc['flux'][cullixs]
    errs = whitelc['uncs'][cullixs]
    sigw = np.median( errs )

    # Initialise a batman object: TODO = THIS MAYBE SHOULD DEPEND ON THE 
    batpar, pmodel = get_batman_object( jd, syspars, ld_type='nonlinear', ld_pars=whitelc['ld_nonlin'] )

    # GP inputs for the white lightcurve fit:
    xvec = np.column_stack( [ ivars['hstphasev'], ivars['wavshift_pixelsv'], ivars['cdcsv'] ] )

    # Re-specify literature epoch to time baseline of data:
    Tmidlit = syspars['Tmid']
    while Tmidlit<jd.min():
        Tmidlit += syspars['P']
    while Tmidlit>jd.max():
        Tmidlit -= syspars['P']

    # Initial guesses for free parameters and Get the log-likelihood function::
    A_init = 0.001
    lniLx_init = np.log( 1 )
    lniLy_init = np.log( 1 )
    lniLz_init = np.log( 1 )
    c0_init = 1
    c1_init = 0
    delT_init = 0
    if ( scanmode=='forward' )*( syspars['tr_type']=='primary' ):
        zfuncs = lnprior_lnlike_primary_forward( jd, t, syspars, Tmidlit, batpar )
        RpRs_init = syspars['RpRs']
        labels = [ 'A', 'lniLx', 'lniLy', 'lniLz', 'c0', 'c1', 'RpRs', 'delT' ]
        initvals = [ A_init, lniLx_init, lniLy_init, lniLz_init, c0_init, c1_init, RpRs_init, delT_init ]
        perturbs = np.array( [ 10e-6, 0.1, 0.1, 0.1, 1e-4, 1e-5, 0.01, 2./60./24. ] )
    elif ( scanmode=='bidirection' )*( syspars['tr_type']=='primary' ):
        zfuncs = lnprior_lnlike_primary_bidirection( jd, t, syspars, Tmidlit, batpar, forward, reverse )
        RpRs_init = syspars['RpRs']
        labels = [ 'A_f', 'lniLx_f', 'lniLy_f', 'lniLz_f', 'c0_f', 'c1_f', \
                   'A_r', 'lniLx_r', 'lniLy_r', 'lniLz_r', 'c0_r', 'c1_r', \
                   'RpRs', 'delT' ]
        initvals = [ A_init, lniLx_init, lniLy_init, lniLz_init, c0_init, c1_init, \
                     A_init, lniLx_init, lniLy_init, lniLz_init, c0_init, c1_init, \
                     RpRs_init, delT_init ]
        perturbs = np.array( [ 10e-6, 0.1, 0.1, 0.1, 1e-4, 1e-5, \
                               10e-6, 0.1, 0.1, 0.1, 1e-4, 1e-5, \
                               0.01, 2./60./24. ] )
    elif ( scanmode=='forward' )*( syspars['tr_type']=='secondary' ):
        zfuncs = lnprior_lnlike_secondary_forward( jd, t, syspars, Tmidlit, batpar )
        SecDepth_init = 0.0
        labels = [ 'A', 'lniLx', 'lniLy', 'lniLz', 'c0', 'c1', 'SecDepth', 'delT' ]
        initvals = [ A_init, lniLx_init, lniLy_init, lniLz_init, c0_init, c1_init, SecDepth_init, delT_init ]
        perturbs = np.array( [ 10e-6, 0.1, 0.1, 0.1, 1e-4, 1e-5, 20e-6, 2./60./24. ] )
    elif ( scanmode=='bidirection' )*( syspars['tr_type']=='secondary' ):
        pdb.set_trace() # not implemented yet
    else:
        pdb.set_trace()
    lnprior = zfuncs[0]
    lnlike = zfuncs[1]
    eval_model = zfuncs[2]
    eval_meanfunc = zfuncs[3]
    neglnpost, lnpost = lnpost_func( lnprior, lnlike )    

    # Run an initial MLE:
    data = ( xvec, flux, sigw )
    pfit = scipy.optimize.fmin( neglnpost, initvals, args=data, maxiter=1e4, xtol=1e-4, ftol=1e-4 )
    ndim = len( pfit )
    print( '\nInitial MLE solution:' )
    for i in range( ndim ):
        print( '  {0} = {1}'.format( labels[i], pfit[i] ) )

    # Initial emcee burn-in to possibly locate better solution:
    p0 = [ np.array( pfit ) + perturbs*np.random.randn( ndim ) for i in range( nwalkers ) ]
    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnpost, args=data )
    print( 'Running burn-in' )
    p1, lnp1, _ = sampler.run_mcmc( p0, nburn1 )
    sampler.reset()

    # Multiple walker group emcee:
    chain_list = []
    lnpost_list = []
    acor_list = []
    for k in range( ngroups ):
        # Run an initial burn-in for each walker group:
        print( 'Running second burn-in' )
        pmax1 = p1[np.argmax(lnp1)]
        p0 = [ pmax1 + perturbs*np.random.randn( ndim ) for i in range( nwalkers ) ]
        p2, lnp2, _ = sampler.run_mcmc( p0, nburn2 )
        sampler.reset()
        # Sample the posterior distribution:
        print( 'Running production' )
        p3, lnp3, _ = sampler.run_mcmc( p2, nsteps )
        print( 'nwalkers = ', nwalkers )
        print( 'nsamples = ', nsteps )
        pmax3 = p3[np.argmax(lnp3)]
        chain_list += [ sampler.flatchain.T ]
        lnpost_list += [ sampler.flatlnprobability ]
        acor_list += [ sampler.get_autocorr_time() ]
    chains = np.dstack( chain_list )
    lnpost_arr = np.column_stack( lnpost_list )
    acor_arr = np.column_stack( acor_list )
    gr_arr = gelman_rubin( chains )
    print( 'Completed.\n' )
    
    # Combined the two sets of walkers:
    chain = np.hstack( chain_list ).T
    lnpost_arr = np.concatenate( lnpost_arr )
    
    # Refine the MLE solution:
    maxix = np.argmax( lnpost_arr )
    maxpar = chain[maxix,:]
    mle_arr = scipy.optimize.fmin( neglnpost, maxpar, args=data, maxiter=1e4 )
    mle = {}
    gr = {}
    acor = {}
    nsamples, npar = np.shape( chain )
    for i in range( npar ):
        mle[labels[i]] = mle_arr[i]
        gr[labels[i]] = gr_arr[i]
        acor[labels[i]] = acor_arr[i,:]

    mle['Tmid'] = Tmidlit + mle['delT'] # best-fit mid-time
    syspars['Tmid'] = mle['Tmid'] 
    if syspars['tr_type']=='primary':
        syspars['RpRs'] = mle['RpRs']
    elif syspars['tr_type']=='secondary':
        syspars['FpFs'] = mle['SecDepth'] #None # todo
    else:
        pdb.set_trace()

    if scanmode=='forward':
        mle_model = mle_model_whitelc_forward( jd, flux, sigw, xvec, mle['Tmid'], mle_arr, chain, \
                                               eval_model, eval_meanfunc, odir )
    elif scanmode=='bidirection':
        mle_model = mle_model_whitelc_bidirection( jd, flux, sigw, xvec, mle['Tmid'],  mle_arr, chain, \
                                                   eval_model, eval_meanfunc, odir, forward, reverse )
    else:
        pdb.set_trace()

    # Corner plot:
    print( 'Making the corner plot...' )
    fig2 = corner.corner( chain, labels=labels )
    opath_fig2 = os.path.join( odir, 'white_corner.pdf' )
    fig2.savefig( opath_fig2 )
    # Sample statistics:
    chprops = chain_properties( chain, labels )
    # Create the output string:
    outstr = '{0}\n# Param  MLE  Med  l34  u34  GR  acor1  acor2\n{1}\n'.format( 50*'#', 50*'-' )
    npar = len( labels )
    for i in range( npar ):
        k = labels[i]
        outstr += '{0} {1} {2} {3} {4} {5:.3f} {6:.0f}\n'\
                  .format( labels[i], mle[k], chprops[k]['med'], chprops[k]['l34'], chprops[k]['u34'], \
                           gr[k], acor[k][0], acor[k][1] )
    print( outstr )
    ofile = open( opath_txt, 'w' )
    ofile.write( outstr )
    ofile.close()
    output = {}
    output['whitelc'] = whitelc
    output['scanmode'] = scanmode
    output['scandirs'] = scandirs
    output['cullixs'] = cullixs
    output['syspars'] = syspars
    output['red'] = red
    output['Tmidlit'] = Tmidlit
    output['walker_chains'] = chain_list
    output['posterior_properties'] = chprops
    output['mle_vals'] = mle
    output['mle_model'] = mle_model
    output['grs'] = gr
    output['acor'] = acor
    ofile = open( opath_mcmc, 'wb' )
    pickle.dump( output, ofile )
    ofile.close()
    # Also save a version without the bulky emcee chains:
    del output['walker_chains']
    ofile = open( opath_mle, 'wb' )
    pickle.dump( output, ofile )
    ofile.close()
    
    print( '\nSaved:\n{0}\n{1}\n{2}\n{3}\n'\
           .format( opath_mcmc, opath_mle, opath_txt, opath_fig2 ) )
    plt.ion()

    return opath_txt, opath_mcmc, opath_mle


def mle_model_whitelc_forward( jd, flux, sigw, xvec, Tmid, mle_arr, chain, eval_model, eval_meanfunc, odir ):
    thrs = ( jd-Tmid )*24
    nsamples, npar = np.shape( chain )
    xeval = xvec
    ttrend_mle, psignal_mle, gpobj_mle = eval_model( mle_arr, xvec )
    gpobj_mle.compute( xvec, yerr=sigw )
    resids_mleb = flux - ttrend_mle*psignal_mle
    mu_mle, cov_mle = gpobj_mle.predict( resids_mleb, xeval )
    std_mle = np.sqrt( np.diag( cov_mle ) )
    zb_mle = ttrend_mle*psignal_mle 
    zc_mle = zb_mle + mu_mle
    resids_mlec = flux - zc_mle
    resids_std = np.std( resids_mlec )
    resids_scatter = resids_std/sigw
    plt.ioff()
    print( '\nPlotting the white lightcurve fit...' )
    fig1 = plt.figure( figsize=[8,10] )
    ax1a = fig1.add_axes( [ 0.12, 0.55, 0.85, 0.42 ] )
    ax1b = fig1.add_axes( [ 0.12, 0.25, 0.85, 0.30 ], sharex=ax1a )
    ax1c = fig1.add_axes( [ 0.12, 0.05, 0.85, 0.20 ], sharex=ax1a )
    ax1a.errorbar( thrs, flux, yerr=sigw, fmt='.k' )
    ax1b.errorbar( thrs, (1e6)*resids_mleb, yerr=(1e6)*sigw, fmt='.k' )
    ax1c.errorbar( thrs, (1e6)*resids_mlec, yerr=(1e6)*sigw, fmt='.k' )
    ndraws = 20
    for ix in np.random.randint( nsamples, size=ndraws ):
        ttrend_i, psignal_i, gpobj_i = eval_model( chain[ix,:] )
        gpobj_i.compute( xvec, yerr=sigw )
        resids = flux - ttrend_mle*psignal_mle
        gpsamp_i = gpobj_i.sample_conditional( resids, xeval )
        ax1a.plot( thrs, ttrend_mle*psignal_mle+gpsamp_i, '-r', alpha=0.3 )
        ax1b.plot( thrs, (1e6)*gpsamp_i, '-r', alpha=0.3 )
    jdf = np.linspace( jd.min(), jd.max(), 500 )
    tf = ( jdf-np.mean( jd ) )/np.std( jd )
    thrsf = ( jdf-Tmid )*24.
    ttrendf, psignalf = eval_meanfunc( jdf, tf, mle_arr[4:] )
    ax1a.plot( thrsf, ttrendf*psignalf, '-g' )
    ax1a.plot( thrs, zc_mle, '-b' )
    ax1b.plot( thrs, (1e6)*mu_mle, '-b' )
    ax1a.fill_between( thrs, zc_mle-std_mle, zc_mle+std_mle, color=0.8*np.ones( 3 ), zorder=0 )
    ax1b.fill_between( thrs, (1e6)*(mu_mle-std_mle), (1e6)*(mu_mle+std_mle), color=0.8*np.ones( 3 ), zorder=0 )
    ax1c.axhline( 0, zorder=0 )
    ax1a.set_ylabel( 'relative flux' )
    ax1b.set_ylabel( 'resids (ppm)' )
    ax1c.set_ylabel( 'resids (ppm)' )
    ax1c.set_xlabel( 'time-from-mid (hrs)' )
    plt.setp( ax1a.xaxis.get_ticklabels(), visible=False )
    plt.setp( ax1b.xaxis.get_ticklabels(), visible=False )
    fig1.suptitle( 'Scatter in residuals = {0:.3f} x photon noise'.format( resids_scatter ) )
    opath_fig1 = os.path.join( odir, 'white_lcfit.pdf' )
    fig1.savefig( opath_fig1 )
    mle_model = { 'psignal':psignal_mle, 'ttrend':ttrend_mle, 'systematics':mu_mle, 'jd':jd, 'thrs':thrs, \
                  'jdf':jdf, 'thrsf':thrsf, 'psignalf':psignalf, 'ttrendf':ttrendf }
    return mle_model


def mle_model_whitelc_bidirection( jd, flux, sigw, xvec, Tmid, mle_arr, chain, \
                                   eval_model, eval_meanfunc, odir, forward, reverse ):
    thrs = ( jd-Tmid )*24
    nsamples, npar = np.shape( chain )
    A_f, lniLx_f, lniLy_f, lniLz_f, c0_f, c1_f = mle_arr[:6] # forward-scan systematics
    A_r, lniLx_r, lniLy_r, lniLz_r, c0_r, c1_r = mle_arr[6:12] # reverse-scan systematics
    RpRs, delT = mle_arr[12:] # planet signal
    meanpars_f = [ c0_f, c1_f, RpRs, delT ]
    meanpars_r = [ c0_r, c1_r, RpRs, delT ]
    xeval = xvec
    ttrend_f, ttrend_r, psignal_f, psignal_r, gpobj_f, gpobj_r = eval_model( mle_arr )
    print( '\nPlotting the white lightcurve fit...' )
    fig1 = plt.figure( figsize=[12,10] )
    ax1a_f = fig1.add_axes( [ 0.12, 0.55, 0.4, 0.42 ] )
    ax1b_f = fig1.add_axes( [ 0.12, 0.25, 0.4, 0.30 ], sharex=ax1a_f )
    ax1c_f = fig1.add_axes( [ 0.12, 0.05, 0.4, 0.20 ], sharex=ax1a_f )
    ax1a_r = fig1.add_axes( [ 0.12+0.45, 0.55, 0.4, 0.42 ], sharex=ax1a_f )
    ax1b_r = fig1.add_axes( [ 0.12+0.45, 0.25, 0.4, 0.30 ], sharex=ax1a_f )
    ax1c_r = fig1.add_axes( [ 0.12+0.45, 0.05, 0.4, 0.20 ], sharex=ax1a_f )
    axs = [ [ ax1a_f, ax1b_f, ax1c_f ], [ ax1a_r, ax1b_r, ax1c_r ] ]
    ttrend = [ ttrend_f, ttrend_r ]
    psignal = [ psignal_f, psignal_r ]
    gpobj = [ gpobj_f, gpobj_r ]
    meanpars = [ meanpars_f, meanpars_r ]
    ixs = [ forward, reverse ]
    ndirect = 2
    mu = []
    jd = [ jd[ixs[0]], jd[ixs[1]] ]
    thrs = [ thrs[ixs[0]], thrs[ixs[1]] ]
    jdf = []
    tf = []
    thrsf = []
    ttrendf = []
    psignalf = []
    for i in range( ndirect ):
        gpobj[i].compute( xvec[ixs[i],:], yerr=sigw )
        residsb_i = flux[ixs[i]] - ttrend[i]*psignal[i]
        mu_i, cov_i = gpobj[i].predict( residsb_i, xeval[ixs[i],:] )
        std_i = np.sqrt( np.diag( cov_i ) )
        zb_i = ttrend[i]*psignal[i]
        zc_i = zb_i + mu_i
        residsc_i = flux[ixs[i]] - zc_i
        residsc_std_i = np.std( residsc_i )
        residsc_scatter_i = residsc_std_i/sigw
        plt.ioff()
        axs[i][0].errorbar( thrs[i], flux[ixs[i]], yerr=sigw, fmt='.k' )
        axs[i][1].errorbar( thrs[i], (1e6)*residsb_i, yerr=(1e6)*sigw, fmt='.k' )
        axs[i][2].errorbar( thrs[i], (1e6)*residsc_i, yerr=(1e6)*sigw, fmt='.k' )
        ndraws = 20
        for ix in np.random.randint( nsamples, size=ndraws ):
            ttrend_fi, ttrend_ri, psignal_fi, psignal_ri, gpobj_fi, gpobj_ri = eval_model( chain[ix,:] )
            gpobj_i = [ gpobj_fi, gpobj_ri ][i]
            gpobj_i.compute( xvec[ixs[i],:], yerr=sigw )
            resids_i = flux[ixs[i]] - ttrend[i]*psignal[i]
            gpsamp_i = gpobj_i.sample_conditional( resids_i, xeval[ixs[i],:] )
            axs[i][0].plot( thrs[i], ttrend[i]*psignal[i]+gpsamp_i, '-r', alpha=0.3 )
            axs[i][1].plot( thrs[i], (1e6)*gpsamp_i, '-r', alpha=0.3 )
        jdf_i = np.linspace( jd[i].min(), jd[i].max(), 500 )
        tf_i = ( jdf_i-np.mean( jd[i] ) )/np.std( jd[i] )
        thrsf_i = ( jdf_i-Tmid )*24.
        ttrendf_i, psignalf_i = eval_meanfunc( jdf_i, tf_i, meanpars[i] )
        axs[i][0].plot( thrsf_i, ttrendf_i*psignalf_i, '-g' )
        axs[i][0].plot( thrs[i], zc_i, '-b' )
        axs[i][1].plot( thrs[i], (1e6)*mu_i, '-b' )
        axs[i][0].fill_between( thrs[i], zc_i-std_i, zc_i+std_i, color=0.8*np.ones( 3 ), zorder=0 )
        axs[i][1].fill_between( thrs[i], (1e6)*(mu_i-std_i), (1e6)*(mu_i+std_i), \
                                color=0.8*np.ones( 3 ), zorder=0 )
        axs[i][2].axhline( 0, zorder=0 )
        axs[i][0].set_title( 'Scatter in residuals = {0:.3f} x photon noise'.format( residsc_scatter_i ) )
        mu += [ mu_i ]
        jdf += [ jdf_i ]
        tf += [ tf_i ]
        thrsf += [ thrsf_i ]
        ttrendf += [ ttrendf_i ]
        psignalf += [ psignalf_i ]
    ax1a_f.set_ylabel( 'relative flux' )
    ax1b_f.set_ylabel( 'resids (ppm)' )
    ax1c_f.set_ylabel( 'resids (ppm)' )
    ax1c_f.set_xlabel( 'time-from-mid (hrs)' )
    ax1c_r.set_xlabel( 'time-from-mid (hrs)' )
    for ax in [ ax1a_f, ax1b_f, ax1a_r, ax1b_r ]:
        plt.setp( ax.xaxis.get_ticklabels(), visible=False )
    for ax in [ ax1a_r, ax1b_r, ax1c_r ]:
        plt.setp( ax.yaxis.get_ticklabels(), visible=False )

    opath_fig1 = os.path.join( odir, 'white_lcfit.pdf' )
    fig1.savefig( opath_fig1 )
    print( 'Saved: {0}'.format( opath_fig1 ) )
    mle_model = { 'psignal':psignal, 'ttrend':ttrend, 'systematics':mu, 'jd':jd, 'thrs':thrs, \
                  'jdf':jdf, 'thrsf':thrsf, 'psignalf':psignalf, 'ttrendf':ttrendf }
    return mle_model



def fit_speclcs_ttrend_quick( whitemle_fpath, speclcs_fpath ):
    """
    Fits the spectroscopic lightcurves with a very simple linear time trend baseline
    multiplied by the transit/eclipse model. The reason for adopting such a simple 
    systematics model is to get a quick transmission/emission spectrum out, without
    using GPs or MCMC. Hopefully this is useful for providing an initial indication
    of the quality and content of a given dataset.
    """

    ifile = open( whitemle_fpath, 'rb' )
    whitemle = pickle.load( ifile )
    ifile.close()
    ifile = open( speclcs_fpath, 'rb' )
    speclcs = pickle.load( ifile )
    ifile.close()

    scandirs = whitemle['scandirs']
    scanmode, forward, reverse = determine_scanmode( scandirs )
    syspars = whitemle['syspars']
    jd = speclcs['jd']
    tv = speclcs['auxvars']['jdv']
    labels = [ 'raw', 'cm1', 'cm2', 'ss' ]
    flux_raw = speclcs['flux_raw']
    flux_cm1 = speclcs['flux_cm1']
    flux_cm2 = speclcs['flux_cm2']
    flux_ss = speclcs['flux_ss']
    uncs_raw = speclcs['uncs_raw']
    uncs_cm1 = speclcs['uncs_cm1']
    uncs_cm2 = speclcs['uncs_cm2']
    uncs_ss = speclcs['uncs_ss']
    flux = [ flux_raw, flux_cm1, flux_cm2, flux_ss ]
    uncs = [ uncs_raw, uncs_cm1, uncs_cm2, uncs_ss ]

    nch, ndat = flux_ss.shape
    ntypes = len( flux )
    psignal = []
    ttrend = []
    nf = 500
    jdf = np.linspace( jd.min(), jd.max(), nf )
    psignalf = []
    ttrendf = []
    
    wavc = np.mean( speclcs['wavedges'], axis=1 )
    RpRs_vals = np.zeros( [ nch, ntypes ] )
    RpRs_uncs = np.zeros( [ nch, ntypes ] )
    for k in range( ntypes ):
        psignalk = np.zeros( [ ndat, nch ] )
        ttrendk = np.zeros( [ ndat, nch ] )
        psignalfk = np.zeros( [ nf, nch ] )
        ttrendfk = np.zeros( [ nf, nch ] )
        print( '' )
        for i in range( nch ):
            fluxki = flux[k][i,:]
            uncski = uncs[k][i,:]
            ldi = speclcs['ld_nonlin'][i,:]
            batpar, pmodel = get_batman_object( jd, syspars, ld_type='nonlinear', ld_pars=ldi )
            if scanmode=='forward':
                model_func_ttrend, initvals = speclcs_ttrend_forward( tv, whitemle, pmodel, batpar, syspars )
            elif scanmode=='bidirection':
                model_func_ttrend, initvals = speclcs_ttrend_bidirection( tv, whitemle, pmodel, batpar, \
                                                                          syspars, forward, reverse )
            else:
                pdb.set_trace()
            pfit = scipy.optimize.curve_fit( model_func_ttrend, jd, fluxki, p0=initvals, \
                                             sigma=uncski, absolute_sigma=True )
            RpRs_val = pfit[0][-1]
            RpRs_unc = np.sqrt( np.diag( pfit[1] ) )[-1]
            RpRs_vals[i,k] = RpRs_val
            RpRs_uncs[i,k] = RpRs_unc
            print( 'Analysis {0:.0f}. Lightcurve {1:.0f}. fitval = {2}'.format( k+1, i+1, pfit[0][-1] ) )
    plt.figure()
    doff = 0.1*np.median( np.diff( wavc ) )
    for k in range( ntypes ):
        plt.errorbar( wavc+k*doff, RpRs_vals[:,k], yerr=RpRs_uncs[:,k], fmt='.-', label=labels[k] )
    plt.legend()
    return wavc, RpRs_vals, RpRs_uncs

def speclcs_ttrend_forward( tv, whitemle, pmodel, batpar, syspars ):
    if syspars['tr_type']=='primary':
        initvals = [ 1, 0, whitemle['mle_vals']['RpRs'] ]
        batpar.t0 = whitemle['mle_vals']['Tmid']
        def model_func_ttrend( jd, c0, c1, RpRs ):
            ttrend = c0 + c1*tv
            batpar.rp = RpRs
            psignal = pmodel.light_curve( batpar )
            return ttrend*psignal
    elif syspars['tr_type']=='secondary':
        initvals = [ 1, 0, whitemle['mle_vals']['SecDepth'] ]
        batpar.t_secondary = whitemle['mle_vals']['Tmid']
        def model_func_ttrend( jd, c0, c1, SecDepth ):
            ttrend = c0 + c1*tv
            batpar.fp = SecDepth
            psignal = pmodel.light_curve( batpar )
            return ttrend*psignal
    else:
        pdb.set_trace() # error with tr_type specification
    return model_func_ttrend, initvals

def speclcs_ttrend_bidirection( tv, whitemle, pmodel, batpar, syspars, forward, reverse ):
    tv_f = tv[forward]
    tv_r = tv[reverse]
    ixs = np.argsort( np.concatenate( [ tv_f, tv_r ] ) )
    if syspars['tr_type']=='primary':
        initvals = [ 1, 0, 1, 0, whitemle['mle_vals']['RpRs'] ]
        batpar.t0 = whitemle['mle_vals']['Tmid']
        def model_func_ttrend( jd, c0_f, c1_f, c0_r, c1_r, RpRs ):
            ttrend_f = c0_f + c1_f*tv[forward]
            ttrend_r = c0_r + c1_r*tv[reverse]
            batpar.rp = RpRs
            psignal = pmodel.light_curve( batpar )
            model_f = ttrend_f*psignal[forward]
            model_r = ttrend_r*psignal[reverse]
            y = np.concatenate( [ model_f, model_r ] )[ixs]
            return y
    elif syspars['tr_type']=='secondary':
        initvals = [ 1, 0, 1, 0, whitemle['mle_vals']['SecDepth'] ]
        batpar.t_secondary = whitemle['mle_vals']['Tmid']
        def model_func_ttrend( jd, c0_f, c1_f, c0_r, c1_r, SecDepth ):
            ttrend_f = c0_f + c1_f*tv[forward]
            ttrend_r = c0_r + c1_r*tv[reverse]
            batpar.fp = SecDepth
            psignal = pmodel.light_curve( batpar )
            model_f = ttrend_f*psignal[forward]
            model_r = ttrend_r*psignal[reverse]
            y = np.concatenate( [ model_f, model_r ] )[ixs]
            return y
    else:
        pdb.set_trace() # error with tr_type specification
    return model_func_ttrend, initvals

#################################################################################
# Likelihood functions.
# Standard likelihood functions for the primary transit and secondary eclipse models.
# As with the prior functions defined above, these could in theory be edited as
# appropriate for the dataset under consideration, but in practice this is hopefully
# not required for the initial quicklook of the transmission/emission spectrum.

def lnprior_lnlike_primary_forward( jd, t, syspars, Tmidlit, batpar ):
    """
    Defines the prior and data likelihood for a primary transit model.
    """
    lnprior = lnprior_primary_forward
    def eval_model_primary( pars ):
        A, lniLx, lniLy, lniLz, c0, c1, RpRs, delT = pars
        ttrend, psignal = eval_meanfunc_primary( jd, t, [ c0, c1, RpRs, delT ] )
        A2 = A**2.
        lniL = np.array( [ lniLx, lniLy, lniLz ] )
        L2 = ( 1./np.exp( lniL ) )**2.
        kernel = A2*george.kernels.Matern32Kernel( L2, ndim=L2.size )
        gpobj = george.GP( kernel )
        return ttrend, psignal, gpobj
    def eval_meanfunc_primary( jd, t, pars ):
        c0, c1, RpRs, delT = pars
        ttrend = c0 + c1*t
        batpar.t0 = Tmidlit + delT
        batpar.rp = RpRs
        pmodel = batman.TransitModel( batpar, jd, fac=0.02, transittype='primary' )
        psignal = pmodel.light_curve( batpar )        
        return ttrend, psignal
    def lnlike( pars, x, y, e ):
        """
        Log( data likelihood ) function
        """
        ttrend, psignal, gpobj = eval_model_primary( pars, x )
        gpobj.compute( x, yerr=e )
        resids = y - ttrend*psignal
        return gpobj.lnlikelihood( resids, quiet=True )
    return lnprior, lnlike, eval_model_primary, eval_meanfunc_primary

def lnprior_lnlike_primary_bidirection( jd, t, syspars, Tmidlit, batpar, forward, reverse ):
    """
    Defines the prior and data likelihood for a primary transit model.
    """
    lnprior = lnprior_primary_bidirection
    def eval_model_primary( pars ):
        # Unpack the parameters:
        A_f, lniLx_f, lniLy_f, lniLz_f, c0_f, c1_f = pars[:6] # forward-scan systematics
        A_r, lniLx_r, lniLy_r, lniLz_r, c0_r, c1_r = pars[6:12] # reverse-scan systematics
        RpRs, delT = pars[12:] # planet signal
        # Evaluate the forward-scan components:
        ttrend_f, psignal_f = eval_meanfunc_primary( jd[forward], t[forward], [ c0_f, c1_f, RpRs, delT ] )
        A2_f = A_f**2.
        lniL_f = np.array( [ lniLx_f, lniLy_f, lniLz_f ] )
        L2_f = ( 1./np.exp( lniL_f ) )**2.
        kernel_f = A2_f*george.kernels.Matern32Kernel( L2_f, ndim=L2_f.size )
        gpobj_f = george.GP( kernel_f )
        # Evaluate the reverse-scan components:
        ttrend_r, psignal_r = eval_meanfunc_primary( jd[reverse], t[reverse], [ c0_r, c1_r, RpRs, delT ] )
        A2_r = A_r**2.
        lniL_r = np.array( [ lniLx_r, lniLy_r, lniLz_r ] )
        L2_r = ( 1./np.exp( lniL_r ) )**2.
        kernel_r = A2_r*george.kernels.Matern32Kernel( L2_r, ndim=L2_r.size )
        gpobj_r = george.GP( kernel_r )
        return ttrend_f, ttrend_r, psignal_f, psignal_r, gpobj_f, gpobj_r
    def eval_meanfunc_primary( jd, t, pars ):
        c0, c1, RpRs, delT = pars
        ttrend = c0 + c1*t
        batpar.t0 = Tmidlit + delT
        batpar.rp = RpRs
        pmodel = batman.TransitModel( batpar, jd, fac=0.02, transittype='primary' )
        psignal = pmodel.light_curve( batpar )        
        return ttrend, psignal
    def lnlike( pars, x, y, e ):
        """
        Log( data likelihood ) function
        """
        ttrend_f, ttrend_r, psignal_f, psignal_r, gpobj_f, gpobj_r = eval_model_primary( pars )
        # Compute log-likelihood for the forward scan:
        gpobj_f.compute( x[forward,:], yerr=e )
        resids_f = y[forward] - ttrend_f*psignal_f
        lnlike_f = gpobj_f.lnlikelihood( resids_f, quiet=True )
        # Compute log-likelihood for the reverse scan:
        gpobj_r.compute( x[reverse,:], yerr=e )
        resids_r = y[reverse] - ttrend_r*psignal_r
        lnlike_r = gpobj_r.lnlikelihood( resids_r, quiet=True )
        return lnlike_f+lnlike_r
    return lnprior, lnlike, eval_model_primary, eval_meanfunc_primary


def lnprior_lnlike_secondary_forward( jd, t, syspars, scanmode, Tmidlit, batpar ):
    """
    Defines the prior and data likelihood for a primary transit model.
    """
    lnprior = lnprior_secondary_forward
    def eval_model_secondary( pars, x ):
        """
        
        """
        A, lniLx, lniLy, lniLz, c0, c1, SecDepth, delT = pars
        ttrend, psignal = eval_meanfunc_secondary( jd, t, [ c0, c1, SecDepth, delT ] )
        A2 = A**2.
        lniL = np.array( [ lniLx, lniLy, lniLz ] )
        L2 = ( 1./np.exp( lniL ) )**2.
        kernel = A2*george.kernels.Matern32Kernel( L2, ndim=L2.size )
        gpobj = george.GP( kernel )
        return ttrend, psignal, gpobj
    def eval_meanfunc_secondary( jd, t, pars ):
        c0, c1, SecDepth, delT = pars
        ttrend = c0 + c1*t
        batpar.t_secondary = Tmidlit + delT
        batpar.fp = SecDepth
        pmodel = batman.TransitModel( batpar, jd, fac=0.02, transittype='secondary' )
        psignal = pmodel.light_curve( batpar )        
        return ttrend, psignal
    def lnlike( pars, x, y, e ):
        """
        Log( data likelihood ) function
        """
        ttrend, psignal, gpobj = eval_model_secondary( pars, x )
        gpobj.compute( x, yerr=e )
        resids = y - ttrend*psignal
        return gpobj.lnlikelihood( resids, quiet=True )
    return lnprior, lnlike, eval_model_secondary, eval_meanfunc_secondary


def lnprior_lnlike_secondary_bidirection( jd, t, syspars, Tmidlit, batpar, forward, reverse ):
    """
    Defines the prior and data likelihood for a primary transit model.
    """
    lnprior = lnprior_secondary_bidirection
    def eval_model_secondary( pars ):
        # Unpack the parameters:
        A_f, lniLx_f, lniLy_f, lniLz_f, c0_f, c1_f = pars[:6] # forward-scan systematics
        A_r, lniLx_r, lniLy_r, lniLz_r, c0_r, c1_r = pars[6:12] # reverse-scan systematics
        SecDepth, delT = pars[12:] # planet signal
        # Evaluate the forward-scan components:
        ttrend_f, psignal_f = eval_meanfunc_secondary( jd[forward], t[forward], [ c0_f, c1_f, SecDepth, delT ] )
        A2_f = A_f**2.
        lniL_f = np.array( [ lniLx_f, lniLy_f, lniLz_f ] )
        L2_f = ( 1./np.exp( lniL_f ) )**2.
        kernel_f = A2_f*george.kernels.Matern32Kernel( L2_f, ndim=L2_f.size )
        gpobj_f = george.GP( kernel_f )
        # Evaluate the reverse-scan components:
        ttrend_r, psignal_r = eval_meanfunc_secondary( jd[reverse], t[reverse], [ c0_r, c1_r, SecDepth, delT ] )
        A2_r = A_r**2.
        lniL_r = np.array( [ lniLx_r, lniLy_r, lniLz_r ] )
        L2_r = ( 1./np.exp( lniL_r ) )**2.
        kernel_r = A2_r*george.kernels.Matern32Kernel( L2_r, ndim=L2_r.size )
        gpobj_r = george.GP( kernel_r )
        return ttrend_f, ttrend_r, psignal_f, psignal_r, gpobj_f, gpobj_r
    def eval_meanfunc_secondary( jd, t, pars ):
        c0, c1, SecDepth, delT = pars
        ttrend = c0 + c1*t
        batpar.t_secondary = Tmidlit + delT
        batpar.fp = SecDepth
        pmodel = batman.TransitModel( batpar, jd, fac=0.02, transittype='secondary' )
        psignal = pmodel.light_curve( batpar )        
        return ttrend, psignal
    def lnlike( pars, x, y, e ):
        """
        Log( data likelihood ) function
        """
        ttrend_f, ttrend_r, psignal_f, psignal_r, gpobj_f, gpobj_r = eval_model_secondary( pars )
        # Compute log-likelihood for the forward scan:
        gpobj_f.compute( x[forward,:], yerr=e )
        resids_f = y[forward] - ttrend_f*psignal_f
        lnlike_f = gpobj_f.lnlikelihood( resids_f, quiet=True )
        # Compute log-likelihood for the reverse scan:
        gpobj_r.compute( x[reverse,:], yerr=e )
        resids_r = y[reverse] - ttrend_r*psignal_r
        lnlike_r = gpobj_r.lnlikelihood( resids_r, quiet=True )
        return lnlike_f+lnlike_r
    return lnprior, lnlike, eval_model_secondary, eval_meanfunc_secondary


  
  def lnpost_func( lnprior, lnlike ):
    """
    Defines the model posterior to be marginalised over.
    """
    def lnpost( pars, x, y, e ):
        l1 = lnprior( pars )
        if np.isfinite( l1 )==False:
            return -np.inf
        else:
            l2 = lnlike( pars, x, y, e )
            return l1 + l2
    def neglnpost( pars, x, y, e ):
        """
        Negative Log( posterior ) function
        """
        z = lnpost( pars, x, y, e )
        return -z
    return neglnpost, lnpost

###############################################################################
# Limb-darkening routines:

def ld_fit_law( grid_mu, grid_wav_nm, grid_intensities, passband_wav_nm, \
                cuton_wav_nm=None, cutoff_wav_nm=None, \
                passband_sensitivity=None, plot_fits=False ):
    """
    Given an ATLAS stellar model grid, computes the limb darkening coefficients
    for four different limb darkening laws: linear, quadratic, three-parameter
    nonlinear and four-parameter nonlinear.
    """    

    # Make sure the throughput goes to zero at the edges:
    ixs = np.argsort( passband_wav_nm )
    passband_wav_nm = passband_wav_nm[ixs]
    passband_sensitivity = passband_sensitivity[ixs]
    dw = 1e-2
    wl = passband_wav_nm.min()-dw
    wu = passband_wav_nm.max()+dw
    passband_wav_nm = np.concatenate( [ [wl], passband_wav_nm, [wu] ] )
    passband_sensitivity = np.concatenate( [ [0], passband_sensitivity, [0] ] )

    # Restrict stellar model to WFC3 wavelength range:
    ixs = ( grid_wav_nm<5e3 )
    grid_wav_nm = grid_wav_nm[ixs]
    grid_intensities = grid_intensities[ixs,:]

    # Interpolate onto a finer grid to allow for narrow channels:
    nf = int( 1e5 )
    xf = np.linspace( grid_wav_nm.min(), grid_wav_nm.max(), nf )
    nmu = len( grid_mu )
    yf = np.zeros( [ nf, nmu ] )
    for i in range( nmu ):
        yf[:,i] = np.interp( xf, grid_wav_nm, grid_intensities[:,i] )
    grid_wav_nm = xf
    grid_intensities = yf    

    # If no passband transmission function has been provided, use
    # a simple boxcar function:
    if passband_sensitivity==None:
        passband_sensitivity = np.ones( passband_wav_nm.size )

    # Interpolate passband wavelengths onto model wavelength grid:
    ixs = np.argsort( passband_wav_nm )
    passband_wav_nm = passband_wav_nm[ixs]
    passband_sensitivity = passband_sensitivity[ixs]
    passband_sensitivity /= passband_sensitivity.max()

    if plot_fits==True:
        stellar_spec = np.mean( grid_intensities, axis=1 )
        fig = plt.figure( figsize=[10,9] )
        ax1 = fig.add_subplot( 211 )
        ax2 = fig.add_subplot( 212 )
        x1 = grid_wav_nm
        y1 = stellar_spec/stellar_spec.max()
        ax1.plot( x1, y1, '-', c='DodgerBlue', label='Mean Stellar Intensity' )
        ax1.fill_between( passband_wav_nm, np.zeros( passband_sensitivity.size ), \
                          passband_sensitivity, edgecolor='Salmon', facecolor='none', zorder=10, label='Passband' )
        ixs1 = ( y1>0.05*y1.max() )*( x1>x1[np.argmax(y1)] )
        ax1.set_xlim( [ 0, x1[ixs1].max() ] )
        ax1.set_xlabel( 'Wavelength (nm)' )
        ax1.set_ylabel( 'Normalised Flux' )
        ax1.legend( loc='upper right' )

    nwav = len( grid_wav_nm )
    mask = np.zeros( nwav )
    ixs = ( grid_wav_nm>=cuton_wav_nm )*\
          ( grid_wav_nm<=cutoff_wav_nm )
    mask[ixs] = 1.0

    interp_sensitivity = np.interp( grid_wav_nm, passband_wav_nm, passband_sensitivity )

    # Integrate the model spectra over the passband for each value of mu:
    nmu = len( grid_mu )
    integrated_intensities = np.zeros( nmu )

    x = grid_wav_nm
    y = grid_wav_nm*mask*interp_sensitivity
    nwav = len( grid_wav_nm )
    ixs = np.arange( nwav )[interp_sensitivity>0]
    ixs = np.concatenate( [ [ ixs[0]-1 ], ixs, [ ixs[-1]+1 ] ] )
    ixs = ixs[(ixs>=0)*(ixs<nwav)]
    normfactor = scipy.integrate.simps( y[ixs], x=x[ixs] )
    #normfactor = scipy.integrate.trapz( y[ixs], x=x[ixs] )
    for i in range( nmu ):
        # Multiply the intensities by wavelength to convert
        # from energy flux to photon flux, as appropriate for
        # photon counting devices such as CCDs:
        integrand = grid_wav_nm*mask*interp_sensitivity*grid_intensities[:,i]
        integral = scipy.integrate.simps( integrand, x=grid_wav_nm )
        integrated_intensities[i] = integral/normfactor
    integrated_intensities /= integrated_intensities[0]

    # Evaluate limb darkening coefficients using linear least
    # squares for each of the four limb darkening laws:
    ld_coeff_fits = {}
    laws = [ fourparam_nonlin_ld, threeparam_nonlin_ld, quadratic_ld, linear_ld ]
    if plot_fits==True:
        ax2.plot( grid_mu, integrated_intensities, 'ok' )
        cs = [ 'PaleGreen', 'ForestGreen', 'Indigo', 'Orange' ]
    for law in laws:
        name, phi = law( grid_mu, coeffs=None )
        # Following Sing (2010), exclude certain values
        # of mu, depending on the limb darkening law:
        if name=='fourparam_nonlin':
            ixs = ( grid_mu>=0 )
        else:
            ixs = ( grid_mu>=0.05 )
        coeffs = np.linalg.lstsq( phi[ixs,:], integrated_intensities[ixs]-1 )[0]
        ld_coeff_fits[name] = coeffs
        if plot_fits==True:
            ax2.plot( grid_mu[ixs], 1+np.dot( phi[ixs,:], coeffs ), '-', c=cs[i], lw=2, label=name )
    if plot_fits==True:
        ax2.legend( loc='upper left' )
        ax2.set_ylabel( 'Passband-integrated Intensity' )
        ax2.set_xlabel( 'mu=cos(theta)' )

    return ld_coeff_fits

def get_bandpass( config ):
    """
    Retrieves the appropriate HST bandpass.
    """
    if config=='G430L':
        # filename should be G430L.sensitivity.sav
        z = idlsave.read( BANDPASS_FPATH_G430L )
        tr_wavs = z['wssens']/10.
        tr_vals = z['sensitivity']
    elif config=='G750L':
        # filename should be G750L.sensitivity.sav
        z = idlsave.read( BANDPASS_FPATH_G750L )
        tr_wavs = z['wssens']/10.
        tr_vals = z['sensitivity']        
    elif config=='G102':
        # filename should be WFC3.IR.G102.1st.sens.2.fits        
        z = pyfits.open( BANDPASS_FPATH_G102 )
        tr_wavs = z[1].data['WAVELENGTH']/10.
        tr_vals = z[1].data['SENSITIVITY']
    elif config=='G141':
        # filename should be WFC3.IR.G141.1st.sens.2.fits        
        z = pyfits.open( BANDPASS_FPATH_G141 )
        tr_wavs = z[1].data['WAVELENGTH']/10.
        tr_vals = z[1].data['SENSITIVITY']
    else:
        pdb.set_trace()  
    return tr_wavs, tr_vals

def get_ld_py( cuton_micron=-1, cutoff_micron=10, config='G141', stagger3d=False ):
    """
    Wrapper for computing the limb darkening coefficients.
    """
    tr_wavs, tr_vals = get_bandpass( config )
    if stagger3d==True:
        pdb.set_trace()
    else:
        mu, wav, intens = read_atlas_grid( model_filepath=ATLAS_FPATH, \
                                           teff=ATLAS_TEFF, logg=ATLAS_LOGG, new_grid=ATLAS_NEWGRID )    
    ld_coeffs = ld_fit_law( mu, wav, intens, tr_wavs, \
                            cuton_wav_nm=cuton_micron*1000, cutoff_wav_nm=cutoff_micron*1000, \
                            passband_sensitivity=tr_vals, plot_fits=False )
    return ld_coeffs

def linear_ld( mus, coeffs=None ):
    """
    Linear limb darkening law.

    I(mu) = c0 - c1*( 1-mu )

    Note, that if coeffs==None, then the basis
    matrix will be returned in preparation for
    finding the limb darkening coeffecients by
    linear least squares. Otherwise, coeffs
    should be an array with 1 entries, one for
    each of the linear limb darkening
    coefficients, in which case the output will
    be the limb darkening law evaluated with
    those coefficients at the locations of the
    mus entries.
    """

    if coeffs==None:
        phi = np.ones( [ len( mus ), 1 ] )
        phi[:,0] = -( 1.0 - mus )

    else:
        phi = 1. - coeffs[1]*( 1.0 - mus )

    return 'linear', phi


def quadratic_ld( mus, coeffs=None ):
    """
    Quadratic limb darkening law.

    I(mu) = c0 - c1*( 1-mu ) - c2*( ( 1-mu )**2. )

    Note, that if coeffs==None, then the basis
    matrix will be returned in preparation for
    finding the limb darkening coeffecients by
    linear least squares. Otherwise, coeffs
    should be an array with 2 entries, one for
    each of the quadratic limb darkening
    coefficients, in which case the output will
    be the limb darkening law evaluated with
    those coefficients at the locations of the
    mus entries.
    """

    if coeffs==None:
        phi = np.ones( [ len( mus ), 2 ] )
        phi[:,0] = -( 1.0 - mus )
        phi[:,1] = -( ( 1.0 - mus )**2. )

    else:
        phi = 1. - coeffs[0]*( 1.0 - mus ) \
              - coeffs[1]*( ( 1.0 - mus )**2. )

    return 'quadratic', phi


def threeparam_nonlin_ld( mus, coeffs=None ):
    """
    The nonlinear limb darkening law as defined
    in Sing 2010.

    Note, that if coeffs==None, then the basis
    matrix will be returned in preparation for
    finding the limb darkening coeffecients by
    linear least squares. Otherwise, coeffs
    should be an array with 4 entries, one for
    each of the nonlinear limb darkening
    coefficients, in which case the output will
    be the limb darkening law evaluated with
    those coefficients at the locations of the
    mus entries.
    """

    if coeffs==None:
        phi = np.ones( [ len( mus ), 3 ] )
        phi[:,0] = - ( 1.0 - mus )
        phi[:,1] = - ( 1.0 - mus**(3./2.) )
        phi[:,2] = - ( 1.0 - mus**2. )

    else:
        phi = 1 - coeffs[0] * ( 1.0 - mus ) \
              - coeffs[1] * ( 1.0 - mus**(3./2.) ) \
              - coeffs[2] * ( 1.0 - mus**2. )

    return 'threeparam_nonlin', phi


def fourparam_nonlin_ld( mus, coeffs=None ):
    """
    The nonlinear limb darkening law as defined
    in Equation 5 of Claret et al 2004.

    Note, that if coeffs==None, then the basis
    matrix will be returned in preparation for
    finding the limb darkening coeffecients by
    linear least squares. Otherwise, coeffs
    should be an array with 4 entries, one for
    each of the nonlinear limb darkening
    coefficients, in which case the output will
    be the limb darkening law evaluated with
    those coefficients at the locations of the
    mus entries.
    """

    if coeffs==None:
        phi = np.ones( [ len( mus ), 4 ] )
        phi[:,0] = - ( 1.0 - mus**(1./2.) )
        phi[:,1] = - ( 1.0 - mus )
        phi[:,2] = - ( 1.0 - mus**(3./2.) )
        phi[:,3] = - ( 1.0 - mus**(2.) )

    else:
        phi = 1 - coeffs[0] * ( 1.0 - mus**(1./2.) ) \
              - coeffs[1] * ( 1.0 - mus ) \
              - coeffs[2] * ( 1.0 - mus**(3./2.) ) \
              - coeffs[3] * ( 1.0 - mus**(2.) )

    return 'fourparam_nonlin', phi

def read_atlas_grid( model_filepath=None, teff=None, logg=None, new_grid=False ):
    """
    Given the full path to an ATLAS model grid, along with values for
    Teff and logg, this routine extracts the values for the specific
    intensity as a function of mu=cos(theta), where theta is the angle
    between the line of site and the emergent radiation. Calling is:

      mu, wav, intensity = atlas.read_grid( model_filepath='filename.pck', \
                                            teff=6000, logg=4.5, vturb=2. )

    Note that the input grids correspond to a given metallicity and
    vturb parameter. So those parameters are controlled by defining
    the model_filepath input appropriately.

    The units of the output variables are:
      mu - unitless
      wav - nm
      intensity - erg/cm**2/s/nm/ster

    Another point to make is that there are some minor issues with the
    formatting of 'new' ATLAS  grids on the Kurucz website. This
    routine will fail on those if you simply download them and feed
    them as input, unchanged. This is because:
      - They have an extra blank line at the start of the file.
      - More troublesome, the last four wavelengths of each grid
        are printed on a single line, which screws up the expected
        structure that this routine requires to read in the file.
    This is 

    """

    # Row dimensions of the input file:
    if new_grid==False:
        nskip = 0 # number of lines to skip at start of file
        nhead = 3 # number of header lines for each grid point
        nwav = 1221 # number of wavelengths for each grid point
    else:
        nskip = 0 # number of lines to skip at start of file
        nhead = 4 # number of header lines for each grid point
        nwav = 1216 # number of wavelengths for each grid point
    nang = 17 # number of angles for each grid point
    # Note: The 'new' model grids don't quite have the 
    # same format, so they won't work for this code.

    print( '\nLimb darkening:\nreading in the model grid...' )
    ifile = open( model_filepath, 'rU' )
    ifile.seek( 0 )
    rows = ifile.readlines()
    ifile.close()
    rows = rows[nskip:]
    nrows = len( rows )
    print( 'Done.' )

    # The angles, where mu=cos(theta):
    mus = np.array( rows[nskip+nhead-1].split(), dtype=float )

    # Read in the teff, logg and vturb values
    # for each of the grid points:
    row_ixs = np.arange( nrows )
    header_ixs = row_ixs[ row_ixs%( nhead + nwav )==0 ]
    if new_grid==True:
        header_ixs += 1
        header_ixs = header_ixs[:-1]
    ngrid = len( header_ixs )
    teff_grid = np.zeros( ngrid )
    logg_grid = np.zeros( ngrid )
    for i in range( ngrid ):
        header = rows[header_ixs[i]].split()
        teff_grid[i] = float( header[1] )
        logg_grid[i] = header[3]

    # Identify the grid point of interest:
    logg_ixs = ( logg_grid==logg )
    teff_ixs = ( teff_grid==teff )

    # Extract the intensities at each of the wavelengths
    # as a function of wavelength:
    grid_ix = ( logg_ixs*teff_ixs )
    row_ix = int( header_ixs[grid_ix] )
    grid_lines = rows[row_ix+nhead:row_ix+nhead+nwav]
    grid = []
    for i in range( nwav ):
        grid += [ grid_lines[i].split() ]
    if new_grid==True:
        grid=grid[:-1]
    grid = np.array( np.vstack( grid ), dtype=float )
    wavs_nm = grid[:,0]
    intensities = grid[:,1:]

    nmus = len( mus )
    for i in range( 1, nmus ):
        intensities[:,i] = intensities[:,i]*intensities[:,0]/100000.

    # Convert the intensities from per unit frequency to
    # per nm in wavelength:
    for i in range( nmus ):
        intensities[:,i] /= ( wavs_nm**2. )    
    return mus, wavs_nm, intensities

#################################################################################
# Utility routines.

def determine_scanmode( scandirs ):
    forward = ( scandirs==1 )
    reverse = ( scandirs==-1 )
    if ( forward.max()==True )*( reverse.max()==True ):
        scanmode = 'bidirection'
    elif ( forward.max()==True  )*( reverse.max()==False ):
        scanmode = 'forward'
    elif ( forward.max()==False  )*( reverse.max()==True ):
        scanmode = 'reverse'
    else:
        pdb.set_trace() # this shouldn't happen
    return scanmode, forward, reverse

def split_orbixs( thrs ):
    tmins = thrs*60.0
    n = len( tmins )
    ixs = np.arange( n )
    dtmins = np.diff( tmins )
    a = 1 + np.arange( n )[dtmins>10]
    a = np.concatenate( [ [0], a, [n] ] )
    norb = len( a ) - 1
    orbixs = []
    for i in range( norb ):
        orbixs += [ np.arange( a[i], a[i + 1] ) ]
    return orbixs

def plot_basic_timeseries( z, red ):
    plt.ioff()
    fig = plt.figure( figsize=[12,8] )
    xlow = 0.12
    axh = ( 1-0.15 )/5.
    axw = 1-0.15
    ax1 = fig.add_axes( [ xlow, 1-0.05-1*axh, axw, axh ] )
    ax2 = fig.add_axes( [ xlow, 1-0.05-2*axh, axw, axh ], sharex=ax1 )
    ax3 = fig.add_axes( [ xlow, 1-0.05-3*axh, axw, axh ], sharex=ax1 )
    ax4 = fig.add_axes( [ xlow, 1-0.05-4*axh, axw, axh ], sharex=ax1 )
    ax5 = fig.add_axes( [ xlow, 1-0.05-5*axh, axw, axh ], sharex=ax1 )
    for ax in [ax2,ax3,ax4,ax5]: plt.setp( ax.xaxis.get_ticklabels(), visible=False )
    c = [ 'c', 'r', 'g', 'm' ]
    for k in range( 4 ):
        jd = z[k]['auxvars']['jd']
        flux = np.sum( z[k]['ecounts'], axis=1 )
        bgppix = z[k]['auxvars']['background_ppix']
        hstphase =  z[k]['auxvars']['hstphase']
        wavshift =  z[k]['auxvars']['wavshift_pixels']
        cdcs =  z[k]['auxvars']['cdcs']
        ax1.plot( jd, flux, '.', c=c[k] )
        ax2.plot( jd, bgppix, '.', c=c[k] )
        ax3.plot( jd, hstphase, '.', c=c[k] )
        ax4.plot( jd, wavshift, '.', c=c[k] )
        ax5.plot( jd, cdcs, '.', c=c[k] )
    ax1.set_ylabel( 'flux' )
    ax2.set_ylabel( 'bgppix' )
    ax3.set_ylabel( 'hstphase' )
    ax4.set_ylabel( 'disp (pix)' )
    ax5.set_ylabel( 'cdisp (pix)' )
    ax5.set_xlabel( 'JD' )
    opath = get_timeseries_fpath( red )
    fig.savefig( opath )
    plt.ion()
    return opath

def get_batman_object( jd, syspars, ld_type=None, ld_pars=[] ):
    # Define the batman planet object:
    batpar = batman.TransitParams()
    batpar.t0 = syspars['T0']
    batpar.per = syspars['P']
    batpar.rp = syspars['RpRs']
    batpar.a = syspars['aRs']
    batpar.inc = syspars['incl']
    batpar.ecc = syspars['ecc']
    batpar.w = syspars['omega']
    batpar.limb_dark = ld_type
    batpar.u = ld_pars
    if syspars['tr_type']=='secondary':
        batpar.fp = syspars['FpFs']
        batpar.t_secondary = syspars['Tmid']
    pmodel = batman.TransitModel( batpar, jd, transittype=syspars['tr_type'] )
    return batpar, pmodel

def get_whitefit_fpath( whitelc_fpath ):
    odir = os.path.join( os.getcwd(), 'results' )
    oname_txt = os.path.basename( whitelc_fpath ).replace( 'whitelc.', 'whitefit.mcmc.' )
    oname_txt = oname_txt.replace( '.pkl', '.txt' )
    opath_txt = os.path.join( odir, oname_txt )
    opath_mcmc = opath_txt.replace( '.txt', '.pkl' )
    opath_mle = opath_mcmc.replace( '.mcmc.', '.mle.' )
    return opath_txt, opath_mcmc, opath_mle

def get_spec_chixs( z, spectra ):
    wav = spectra['wavsol_micron']
    cuton = z['cuton_micron']
    npb = z['npix_perbin']
    nch = z['nchannels']
    ndisp = wav.size
    edges0 = np.arange( ndisp )[np.argmin( np.abs( wav-cuton ) )]
    edges = np.arange( edges0, edges0+( nch+1 )*npb, npb, dtype=int )
    chixs = np.zeros( [ nch, 2 ], dtype=int )
    for i in range( nch ):
        chixs[i,:] = np.array( [ edges[i], edges[i+1] ] )
    return chixs

def get_speclcs_fpath( spectra_fpath ):
    odir = os.path.join( os.getcwd(), 'lightcurves' )
    oname = os.path.basename( spectra_fpath ).replace( 'spectra.', 'speclcs.' )
    opath = os.path.join( odir, oname )
    return opath

def get_whitelc_fpath( spectra_fpath ):
    ofilename = os.path.basename( spectra_fpath ).replace( 'spectra.', 'whitelc.' )
    odir = os.path.join( os.getcwd(), 'lightcurves' )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    opath = os.path.join( odir, ofilename )
    return opath

def get_cullixs( jd, discard_first_exposure ):
    cullixs = discard_first_orbit_ixs( jd )
    if discard_first_exposure==True:
        dt = np.diff( jd[cullixs] )
        cullixs = cullixs[1+np.arange( cullixs.size )[dt<10./60./24.]]
    return cullixs

def get_spectra_fpath( red ):
    suffix = [ '.rlast.pkl', '.rlast.zapped.pkl', '.rdiff.pkl', '.rdiff.zapped.pkl' ]
    opaths = []
    for k in range( 4 ):
        ofilename = 'spectra.aprad{0:.0f}.maskrad{1:.0f}{2}'\
                    .format( red['apradius'], red['maskradius'], suffix[k] )
        odir = os.path.join( os.getcwd(), 'spectra' )
        if os.path.isdir( odir )==False:
            os.makedirs( odir )
        opaths += [ os.path.join( odir, ofilename ) ]
    return opaths

def get_timeseries_fpath( red ):
    ofilename = 'timeseries.aprad{0:.0f}.maskrad{1:.0f}.pdf'\
                .format( red['apradius'], red['maskradius'] )
    odir = os.path.join( os.getcwd(), 'lightcurves' )
    if os.path.isdir( odir )==False:
        os.makedirs( odir )
    opath = os.path.join( odir, ofilename )
    return opath

def get_wavsol( config, flux, make_plot=False ):
    A2micron = 1e-4
    nm2micron = 1e-3
    ndisp = flux.size
    if config=='G141':
        dispersion_nmpix = 0.5*( 4.47+4.78 ) # nm/pixel
    elif config=='G102':
        dispersion_nmpix = 0.5*( 2.36+2.51 ) # nm/pixel
    else:
        pdb.set_trace()
    dispersion = nm2micron*dispersion_nmpix # micron/pixel
    tr_wavs, tr_vals = get_bandpass( config )
    tr_wavs = tr_wavs*nm2micron
    ixs = np.argsort( tr_wavs )
    tr_wavs = tr_wavs[ixs]
    tr_vals = tr_vals[ixs]
    mu, wav, intens = read_atlas_grid( model_filepath=ATLAS_FPATH, teff=ATLAS_TEFF, \
                                       logg=ATLAS_LOGG, new_grid=ATLAS_NEWGRID )
    # Interpolate the stellar model onto the transmission wavelength grid:
    wmodel = wav*nm2micron
    fmodel = intens[:,0]
    ixs = ( wmodel>tr_wavs[0]-0.1 )*( wmodel<tr_wavs[-1]+0.1 )
    fmodel_interp = np.interp( tr_wavs, wmodel[ixs], fmodel[ixs] )
    # Modulate the interpolated stellar model by the throughput to 
    # simulate a measured spectrum:
    fmodel = fmodel_interp*tr_vals
    fmodel /= fmodel.max()
    wmodel = tr_wavs
    ix = np.argmax( fmodel )
    w0 = wmodel[ix]
    x = np.arange( ndisp )
    ix = np.argmax( flux )
    delx = x-x[ix]
    wavsol0 = w0 + dispersion*delx
    # Smooth the stellar flux and model spectrum, because we use
    # the sharp edges of the throughput curve to calibrate the 
    # wavelength solution:
    fwhm = 4. # stdv of smoothing kernel
    smoothing_sig =fwhm/2./np.sqrt( 2.*np.log( 2 ) )
    flux_smooth = scipy.ndimage.filters.gaussian_filter1d( flux, smoothing_sig )
    fmodel_smooth = scipy.ndimage.filters.gaussian_filter1d( fmodel, smoothing_sig )
    # The amount we need to buffer the shifting model spectrum by
    # so that we can trial the desired range of shifts for each
    # measured spectrum:
    dw = np.median( np.diff( wmodel ) )
    dwav_max = 0.3 # micron
    wlow = wavsol0.min()-dwav_max-dw
    wupp = wavsol0.max()+dwav_max+dw
    #nbuff = int( np.ceil( dwav_max/float( dw ) ) )
    # Extend the model spectrum at both edges:
    dwlow = np.max( [ wmodel.min()-wlow, 0 ] )
    dwupp = np.max( [ wupp-wmodel.max(), 0 ] )
    wbuff_lhs = np.r_[ wmodel.min()-dwlow:wmodel.min():dw ]
    wbuff_rhs = np.r_[ wmodel.max()+dw:wmodel.max()+dwupp:dw ]
    wmodel_ext = np.concatenate( [ wbuff_lhs, wmodel, wbuff_rhs ] )
    fbuff_lhs = np.zeros( len( wbuff_lhs ) )
    fbuff_rhs = np.zeros( len( wbuff_rhs ) )
    fmodel_ext = np.concatenate( [ fbuff_lhs, fmodel, fbuff_rhs ] )
    # Interpolate the extended model spectrum:
    interpf = scipy.interpolate.interp1d( wmodel_ext, fmodel_ext )
    nshifts = int( np.round( 2*dwav_max*(1e4)+1 ) ) # 0.00001 micron = 0.1 nm
    shifts = np.r_[-dwav_max:dwav_max:1j*nshifts] # shifts
    rms = np.zeros( nshifts )
    # Loop over the wavelength shifts, where for each shift we move
    # the model spectrum and compare it to the measured spectrum
    A = np.ones( [ ndisp, 2 ] )
    b = np.reshape( flux_smooth/flux_smooth.max(), [ ndisp, 1 ] )
    for i in range( nshifts ):
        # Assuming the default wavelength solution is wavsol0, shift
        # the model spectrum by dx. If this provides a good match to 
        # the data, it means that the default wavelength solution wavsol0
        # is off by an amount dx.
        fmodel_smooth_shifted = interpf( wavsol0 + shifts[i] )
        A[:,1] = fmodel_smooth_shifted/fmodel_smooth_shifted.max()
        res = np.linalg.lstsq( A, b )
        c = res[0].flatten()
        fit = np.dot( A, c )
        diffs = b.flatten() - fit.flatten()
        rms[i] = np.sqrt( np.mean( diffs**2. ) )
    # Locate the shift that gave the best match:
    ix = np.argmin( rms )
    wavsol_crosscorr = wavsol0 + shifts[ix]
    # Set make_plot to True if you want to make a plot to check 
    # the wavelength solution:
    if make_plot==True:
        plt.figure( figsize=[12,8] )
        plt.plot( wavsol_crosscorr, flux/flux.max(), '-m', lw=2, label='cross-correlation' )
        plt.plot( tr_wavs, tr_vals/tr_vals.max(), '-g', label='G141' )
        plt.plot( wav*nm2micron, intens[:,0]/intens[:,0].max(), '-r', label='stellar' )
        plt.plot( wmodel, fmodel/fmodel.max(), '--c', lw=2, label='model spec' )
        plt.xlim( [ 0.8, 2.0 ] ) 
        plt.ylim( [ -0.1, 1.4 ] )
        plt.legend( loc='upper left', ncol=2 )
        pdb.set_trace()
    return wavsol_crosscorr

def get_frames( red ):
    ddir = red['ddir'] 
    ntrim = red['ntrim_edge']
    apradius = red['apradius']
    maskradius = red['maskradius']
    smoothing_fwhm = red['smoothing_fwhm']
    crossdisp_bound_ixs = np.array( red['crossdisp_bound_ixs'] )-ntrim
    trim_disp_ixs = np.array( red['trim_disp_ixs'] )-ntrim
    shiftstretch_disp_ixs = np.array( red['shiftstretch_disp_ixs'] )-ntrim
    bgcd_ixs = np.array( red['bg_crossdisp_ixs'] )-ntrim
    bgd_ixs = np.array( red['bg_disp_ixs'] )-ntrim
    if red['config']=='G141':
        filter_str = 'G141'
    elif red['config']=='G102':
        filter_str = 'G102'
    else:
        pdb.set_trace()

    c1, c2 = crossdisp_bound_ixs
    d1, d2 = trim_disp_ixs

    # Read in the raw frames:
    search_str = os.path.join( ddir, '*_ima.fits' )
    fs = np.array( glob.glob( search_str ), dtype=str )
    nframes = len( fs) 
    tstarts = []
    exptimes = []
    ecounts2d_rlast = [] # ecounts extracted from the last read
    ecounts2d_rdiff = [] # ecounts extracted from the read differences
    abs_m = []
    v = []
    bg_ppix_first_arr = []
    bg_ppix_last_arr = []
    scandirs = []
    for i in range( nframes ):
        hdu = pyfits.open( fs[i] )
        header0 = hdu[0].header
        header1 = hdu[1].header
        if ( header0['OBSTYPE']=='SPECTROSCOPIC' )*( header0['FILTER']==filter_str ):
            print( '... {0} of {1} - keeping {2}+{3}'#
                   .format( i+1, nframes, header0['OBSTYPE'], header0['FILTER'] ) )
            header0 = hdu[0].header
            header1 = hdu[1].header
            frame = hdu[1].data[ntrim:-ntrim,ntrim:-ntrim]
            nscan, ndisp = np.shape( frame )
            c1 = max( [ 0, c1 ] )
            c2 = min( [ nscan, c2 ] )
            d1 = max( [ 0, d1 ] )
            d2 = min( [ ndisp, d2 ] )
            abs_m += [ np.abs( header1['LTM1_1'] ) ]
            v += [ header1['LTV1'] ]
            tstart = header0['EXPSTART']
            exptime = header0['EXPTIME']
            tstarts += [ tstart ]
            exptimes += [ exptime ]
            # Also take the individual reads and difference them:
            nreads = int( ( len( hdu )-1 )/5 )
            nreads -= 1 # exclude the zeroth read
            ix0 = 1+(nreads-1)*5
            first_read = hdu[ix0].data[ntrim:-ntrim,ntrim:-ntrim]
            last_read = hdu[1].data[ntrim:-ntrim,ntrim:-ntrim]
            sampt_first = hdu[ix0].header['SAMPTIME']
            sampt_last = hdu[1].header['SAMPTIME']
            if hdu[1].header['BUNIT'].lower()=='electrons':
                # units are electrons for more recent datasets
                first_read_ecounts = first_read 
                last_read_ecounts = last_read # units now electrons
            elif hdu[1].header['BUNIT'].lower()=='electrons/s':
                # units used to be electrons/sec
                first_read_ecounts = first_read*sampt_first 
                last_read_ecounts = last_read*sampt_last
            else:
                pdb.set_trace()
            # Estimate the background in e/pixel for the first read and remove it:
            bg_box = [ max( [ bgcd_ixs[0], 0 ] ), min( [ bgcd_ixs[1], nscan ] ), \
                       max( [ bgd_ixs[0], 0 ] ), min( [ bgd_ixs[1], ndisp ] ) ]
            bg_ppix_first = np.median( first_read_ecounts[bg_box[0]:bg_box[1],bg_box[2]:bg_box[3]] )
            first_read_ecounts -= bg_ppix_first
            bg_ppix_last = np.median( last_read_ecounts[bg_box[0]:bg_box[1],bg_box[2]:bg_box[3]] )
            ecounts2d_rlast += [ last_read_ecounts - bg_ppix_last ]
            bg_ppix_first_arr += [ bg_ppix_first ]
            bg_ppix_last_arr += [ bg_ppix_last ]
            # Build up the differences between reads:
            ndiffs = nreads-1
            rdiff_ecounts = np.zeros( [ nscan, ndisp, ndiffs ] )
            rdiff_cscans = np.zeros( ndiffs )
            for j in range( ndiffs ):
                # First read:
                ix1 = 1+(nreads-1)*5-j*5
                read1 = hdu[ix1].data[ntrim:-ntrim,ntrim:-ntrim]#[c1:c2+1,d1:d2+1]
                sampt1 = hdu[ix1].header['SAMPTIME']
                # Second read:
                ix2 = 1+(nreads-1)*5-(j+1)*5
                read2 = hdu[ix2].data[ntrim:-ntrim,ntrim:-ntrim]#[c1:c2+1,d1:d2+1]
                sampt2 = hdu[ix2].header['SAMPTIME']
                # Get as electrons:
                if hdu[1].header['BUNIT'].lower()=='electrons':
                    ecounts1 = read1
                    ecounts2 = read2
                elif hdu[1].header['BUNIT'].lower()=='electrons/s':
                    ecounts1 = read1*sampt1
                    ecounts2 = read2*sampt2
                else:
                    pdb.set_trace()
                # Need to perform sky subtraction here to calibrate
                # the flux level between reads, because the sky
                # actually varies quite a lot between successive reads:
                bg1 = np.median( ecounts1[bg_box[0]:bg_box[1],bg_box[2]:bg_box[3]] )
                ecounts1 -= bg1
                bg2 = np.median( ecounts2[bg_box[0]:bg_box[1],bg_box[2]:bg_box[3]] )
                ecounts2 -= bg2
                rdiff_ecounts[:,:,j] = ecounts2 - ecounts1

                # Estimate the center of the scan for purpose of applying mask:
                x = np.arange( nscan )[c1:c2+1]
                ninterp = 1000
                nf = int( ninterp*len( x ) )
                xf = np.r_[x.min():x.max():1j*nf]
                cdp = np.sum( rdiff_ecounts[:,:,j][c1:c2+1,d1:d2+1], axis=1 )
                cdpf = np.interp( xf, x, cdp )
                thresh = cdp.min() + 0.05*( cdp.max()-cdp.min() )
                ixs = ( cdpf>thresh )
                cscan = np.mean( xf[ixs] ) # scan center
                # Apply the mask:
                ixl = int( np.floor( cscan - maskradius ) )
                ixu = int( np.ceil( cscan + maskradius ) )
                rdiff_ecounts[:ixl+1,:,j] = 0.0
                rdiff_ecounts[ixu:,:,j] = 0.0
                rdiff_cscans[j] = cscan
            dscan = rdiff_cscans[-1]-rdiff_cscans[0]
            if dscan>0:
                scandirs += [ +1 ]
            else:
                scandirs += [ -1 ]
            # Construct the final scan, noting that first_read_electrons should
            # already be background-subtracted:
            ecounts_per_read = np.dstack( [ first_read_ecounts, rdiff_ecounts ] )
            ecounts2d_rdiff += [ np.sum( ecounts_per_read, axis=2 ) ]
            hdu.close()
        else: # it must be the acquisition image
            print( '... {0} of {1} - skipping {2}+{3}'\
                   .format( i+1, nframes, header0['OBSTYPE'], header0['FILTER'] ) )
            hdu.close()
            continue
    ecounts2d_rlast = np.dstack( ecounts2d_rlast )
    ecounts2d_rdiff = np.dstack( ecounts2d_rdiff )
    tstarts = np.array( tstarts )
    scandirs = np.array( scandirs )
    exptimes = np.array( exptimes )
    bg_ppix = np.array( bg_ppix_last_arr )
    return ecounts2d_rlast, ecounts2d_rdiff, tstarts, exptimes, scandirs, bg_ppix, fs

def discard_first_orbit_ixs( jd ):
    nframes = len( jd )
    difft = np.diff( jd )*24*60
    ixs = np.arange( nframes )[difft>10]
    ixs = np.arange( int( ixs.min()+1 ), nframes )
    return ixs

def logp_mvnormal_whitenoise( r, u, n  ):
    """
    Log likelihood of a multivariate normal distribution
    with diagonal covariance matrix.
    """
    term1 = -np.sum( np.log( u ) )
    term2 = -0.5*np.sum( ( r/u )**2. )
    return term1 + term2 - 0.5*n*np.log( 2*np.pi )

def gelman_rubin( chains ):
    """
    Routine for Gelman-Rubin convergence statistic.
    """
    npar, nsamples, nchains = np.shape( chains )
    m = nchains
    n = nsamples
    gr = np.zeros( npar )
    for i in range( npar ):
        chain_i = chains[i,:,:]
        W = np.mean( np.var( chain_i, axis=0, ddof=1 ) )
        B_over_n = np.var( np.mean( chain_i, axis=0 ), ddof=1 )
        sigma2 = ( ( n-1. )/n )*W + B_over_n
        Vhat = sigma2 + B_over_n/float( m )
        gr[i] = np.sqrt( Vhat/float( W ) )
    return gr

def chain_properties( chain, labels ):
    nsamples, npar = np.shape( chain )
    n34 = int( np.round( 0.34*nsamples ) )
    properties = {}
    z = {}
    for i in range( npar ):
        k = labels[i]
        z[k] = {}
        z[k]['med'] = np.median( chain[:,i] )
        deltas = chain[:,i] - z[k]['med']
        ixsl = ( deltas<0 )
        ixsu = ( deltas>0 )
        z[k]['l34'] = np.abs( deltas[ixsl][np.argsort( deltas[ixsl] )][-n34] )
        z[k]['u34'] = np.abs( deltas[ixsu][np.argsort( deltas[ixsu] )][n34] )
    return z

###############################################################################
