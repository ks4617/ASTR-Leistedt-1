# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:00:04 2020
"""
#%% from https://www.astroml.org/examples/datasets/compute_sdss_pca.html

import sys
from urllib.error import HTTPError
import numpy as np
from astroML.datasets import fetch_sdss_spectrum
from astroML.datasets.tools import query_plate_mjd_fiber, TARGET_GALAXY
from astroML.dimensionality import iterative_pca
import matplotlib.pyplot as plt
import inspect
#%% Fetch spectra, correct for redshift and rebin wavelengths


def fetch_and_shift_spectra(n_spectra,
                            outfile,
                            primtarget=TARGET_GALAXY,
                            zlim=(0, 0.7),
                            loglam_start=3.5,
                            loglam_end=3.9,
                            Nlam=1000):
    """
    This function queries CAS for matching spectra, and then downloads
    them and shifts them to a common redshift binning
    """
    # First query for the list of spectra to download
    plate, mjd, fiber = query_plate_mjd_fiber(n_spectra, primtarget,
                                              zlim[0], zlim[1])
    print('plate:', plate[0],'mjd:', mjd[0], 'fiber:', fiber[0])                    # Print identifiers to check against other script


    # Set up arrays to hold information gathered from the spectra
    spec_cln = np.zeros(n_spectra, dtype=np.int32)
    lineindex_cln = np.zeros(n_spectra, dtype=np.int32)
    #subclass = np.zeros(n_spectra, dtype=np.int32)

    log_NII_Ha = np.zeros(n_spectra, dtype=np.float32)
    log_OIII_Hb = np.zeros(n_spectra, dtype=np.float32)

    z = np.zeros(n_spectra, dtype=np.float32)
    zerr = np.zeros(n_spectra, dtype=np.float32)
    spectra = np.zeros((n_spectra, Nlam), dtype=np.float32)
    spec_err = np.zeros((n_spectra, Nlam), dtype=np.float32)
    mask = np.zeros((n_spectra, Nlam), dtype=np.bool)

    # Calculate new wavelength coefficients
    new_coeff0 = loglam_start
    new_coeff1 = (loglam_end - loglam_start) / Nlam

    # Now download all the needed spectra, and resample to a common
    #  wavelength bin.
    n_spectra = len(plate)
    num_skipped = 0
    i = 0

    while i < n_spectra:
        sys.stdout.write(' %i / %i spectra\r' % (i + 1, n_spectra))
        sys.stdout.flush()
        try:
            spec = fetch_sdss_spectrum(plate[i], mjd[i], fiber[i])
            
        except HTTPError:
            num_skipped += 1
            print("%i, %i, %i not found" % (plate[i], mjd[i], fiber[i]))
            i += 1
            continue

        spec_rebin = spec.restframe().rebin(new_coeff0, new_coeff1, Nlam) # Redshift corrected

#        spec_rebin = spec.rebin(new_coeff0, new_coeff1, Nlam)   # Not redshift corrected


        if np.all(spec_rebin.spectrum == 0):
            num_skipped += 1
            print("%i, %i, %i is all zero" % (plate[i], mjd[i], fiber[i]))
            i += 1
            continue
        
              
        spec_cln[i] = spec.spec_cln
#        subclass[i] = spec.subclass
        
        lineindex_cln[i], (log_NII_Ha[i], log_OIII_Hb[i])\
            = spec.lineratio_index()

        z[i] = spec.z
        zerr[i] = spec.zerr

        spectra[i] = spec_rebin.spectrum
        spec_err[i] = spec_rebin.error
        mask[i] = spec_rebin.compute_mask(0.5, 5)

        i += 1
    sys.stdout.write('\n')

    N = i
 #   print("   %i spectra skipped" % num_skipped)
 #   print("   %i spectra processed" % N)
 #   print("saving to %s" % outfile)

    np.savez(outfile,
             spectra=spectra[:N],
             spec_err=spec_err[:N],
             mask=mask[:N],
             coeff0=new_coeff0,
             coeff1=new_coeff1,
             spec_cln=spec_cln[:N],
             lineindex_cln=lineindex_cln[:N],
             log_NII_Ha=log_NII_Ha[:N],
             log_OIII_Hb=log_OIII_Hb[:N],
             z=z[:N],
             zerr=zerr[:N],
             plate=plate[:N],
             mjd=mjd[:N],
             fiber=fiber[:N])


def spec_iterative_pca(outfile, n_ev=10, n_iter=20, norm='L2'):
    """
    This function takes the file outputted above, performs an iterative
    PCA to fill in the gaps, and appends the results to the same file.
    """
    data_in = np.load(outfile)
    spectra = data_in['spectra']
    mask = data_in['mask']

    res = iterative_pca(spectra, mask,
                        n_ev=n_ev, n_iter=n_iter, norm=norm,
                        full_output=True)

    input_dict = {key: data_in[key] for key in data_in.files}

    # don't save the reconstructed spectrum: this can easily
    # be recomputed from the other parameters.
    input_dict['mu'] = res[1]
    input_dict['evecs'] = res[2]
    input_dict['evals'] = res[3]
    input_dict['norms'] = res[4]
    input_dict['coeffs'] = res[5]

    np.savez(outfile, **input_dict)


if __name__ == '__main__':
    fetch_and_shift_spectra(30000, 'spec30000mjd.npz')
    #spec_iterative_pca('spec10.npz')
    
#%% Load saved compressed file
spec_one = np.load('spec1test.npz')

#%%
spectra_all = np.load('spec4000.npz')

#%% Create a wavelength array from wavelength coefficients and plot spectra
n = np.arange(0, 1000, 1)
bin_pos = spectra_one['coeff0'] + spectra_one['coeff1']*n

#plt.figure()
plt.plot(bin_pos,spectra_one['spectra'][0], label='Redshift included')
#plt.plot(bin_pos,spectra_one['spectra'][4], label='Redshift corrected by restframe')
#plt.plot(bin_pos-np.log10(1+spectra_one['z'][4]),spectra_one['spectra'][4], label='Redshift corrected by hand')
plt.plot(bin_pos,spectra_one['spec_err'][0],'g', label='error')

plt.xlabel(r'log $\lambda (\AA)$')
plt.ylabel('Flux')
plt.legend()
plt.show()

#%% Load 9 spectra or 18 to get another set of 9
spectra_nine = np.load('spec9.npz')

#%%
spectra_18 = np.load('spec18.npz')

#%%  Nine spectra plot adapted from https://www.astroml.org/book_figures/chapter7/fig_spec_examples.html

n = np.arange(0, 1000, 1)
bin_pos = spectra_nine['coeff0'] + spectra_nine['coeff1']*n

fig = plt.figure(figsize=(5, 4))

fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05,
                    bottom=0.1, top=0.95, hspace=0.05)

ncols = 3
nrows = 3

for i in range(ncols):
    for j in range(nrows):
        ax = fig.add_subplot(nrows, ncols, ncols * j + 1 + i)
        ax.plot(bin_pos, spectra_nine['spectra'][ncols * j + i], '-k', lw=1)
#        ax.plot(bin_pos, spectra_nine['spectra'][9 + ncols * j + i], '-k', lw=1)
        #ax.set_xlim(3, 4)

        ax.yaxis.set_major_formatter(plt.NullFormatter())
        if j < nrows - 1:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        else:
            plt.xlabel(r'log $\lambda (\AA)$')

        ylim = ax.get_ylim()
        dy = 0.05 * (ylim[1] - ylim[0])
        ax.set_ylim(ylim[0] - dy, ylim[1] + dy)

plt.show()

#%% Plot to show comparison of redshift corrected and included
n = np.arange(0, 1000, 1)
bin_pos = spectra_nine['coeff0'] + spectra_nine['coeff1']*n

#plt.figure()
plt.plot(bin_pos,spectra_nine['spectra'][4], label='Redshift corrected')
#plt.plot(bin_pos,spectra_nine['spectra'][4], label='Redshift included')

plt.xlabel(r'log $\lambda (\AA)$')
plt.ylabel('Flux')
plt.legend()
plt.show()

#%% Plot of mask for spectra 5 which has a obvious section masked
spec_11 = spectra_18['mask'][10]
n = np.arange(0, 1000, 1)
bin_pos = spectra_18['coeff0'] + spectra_18['coeff1']*n

plt.figure()
plt.plot(bin_pos,spec_11*3, label='Mask')
plt.plot(bin_pos,spectra_18['spectra'][10], label='Spectra with redshift corrected')

plt.xlabel(r'log $\lambda (\AA)$')
plt.ylabel('Flux')
plt.legend()
plt.show()

#%%
plt.figure()
plt.xticks(np.arange(1,19))
plt.plot(np.arange(1,19),spectra_18['z'])
plt.xlabel('Spectra Number')
plt.ylabel('Redshift z')
plt.show()




