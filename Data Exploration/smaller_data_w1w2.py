import numpy as np
from astropy.table import Table, hstack
# %% Read in data
north = Table.read('survey-dr8-north-specObj-dr14.fits',format='fits')
south = Table.read('survey-dr8-south-specObj-dr14.fits',format='fits')
spectro = Table.read('specObj-dr14.fits',format='fits')

# %% Merge north and south
if True:
    ind = south['OBJID'] == -1
    south[ind] = north[ind]

photo = south


# %% Correct fluxes
bands = np.array(['G','R','Z','W1','W2'])
def correct_flux(table, photo_bands):
    for i in photo_bands:
        table['FLUX_'+i] = table['FLUX_'+i]/table['MW_TRANSMISSION_'+i]
        table['FLUX_IVAR_'+i] = table['FLUX_IVAR_'+i] * (table['MW_TRANSMISSION_'+i]**2)

correct_flux(photo, bands)

# %% Drop unnecessary columns
photo_impt = np.array(['RA','DEC','OBJID'])
spectro_impt = np.array(['MJD','PLATE','FIBERID','Z','ZWARNING','CLASS','SUBCLASS'])
for i in bands:
    photo_impt = np.append(photo_impt,'FLUX_'+i)
    photo_impt = np.append(photo_impt,'FLUX_IVAR_'+i)
    photo_impt = np.append(photo_impt,'MW_TRANSMISSION_'+i)

photo_impt_ebv = np.append(photo_impt, 'EBV')
photo_impt_list = photo_impt.tolist()
photo_impt_list_ebv = photo_impt_ebv.tolist()
spectro_impt_list = spectro_impt.tolist()
photo_sub = photo[photo_impt_list]
photo_sub_ebv = photo[photo_impt_list_ebv]
spectro_sub = spectro[spectro_impt_list]

# %% Save photo and spectro
photo_sub.write('photo_w1w2.fits', format = 'fits')
photo_sub_ebv.write('photo_ebv_w1w2.fits', format = 'fits')
spectro_sub.write('spectro_w1w2.fits', format = 'fits')

# %% Combine photo and spectro
full = hstack([spectro_sub, photo_sub])
full_ebv = hstack([spectro_sub, photo_sub_ebv])

# %% Keep only ZWARNING == 0
full = full[full['ZWARNING'] == 0]
full_ebv = full_ebv[full_ebv['ZWARNING'] == 0]

# %%
full.write('full_w1w2.fits', format = 'fits')
full_ebv.write('full_ebv_w1w2.fits', format = 'fits')