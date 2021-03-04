import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table, hstack, vstack

# %% Read in data
north = Table.read('survey-dr8-north-specObj-dr14.fits',format='fits')
south = Table.read('survey-dr8-south-specObj-dr14.fits',format='fits')
spectro = Table.read('specObj-dr14.fits',format='fits')

# %% Merge north and south
if True:
    ind = south['OBJID'] == -1
    south[ind] = north[ind]

photo = south

# %% Combine photo and spectro
full = hstack([spectro, photo])

# %% Correct fluxes
bands = np.array(['G','R','Z','W1','W2', 'W3', 'W4'])
def correct_flux(table, photo_bands):
    for i in photo_bands:
        table['FLUX_'+i] = table['FLUX_'+i]/table['MW_TRANSMISSION_'+i]
        table['FLUX_IVAR_'+i] = table['FLUX_IVAR_'+i] * (table['MW_TRANSMISSION_'+i]**2)

correct_flux(full, bands)

# %%
full.rename_column('PLATE_1', 'PLATE')
full.rename_column('MJD_1', 'MJD')
full.rename_column('FIBERID_1', 'FIBERID')
    
# %% Drop unnecessary columns
impt_cols = np.array(['MJD','PLATE','FIBERID','RA','DEC','Z','ZWARNING','CLASS', 'SUBCLASS'])
for i in bands:
    impt_cols = np.append(impt_cols,'FLUX_'+i)
    impt_cols = np.append(impt_cols,'FLUX_IVAR_'+i)
    impt_cols = np.append(impt_cols,'MW_TRANSMISSION_'+i)

impt_cols_list = impt_cols.tolist()
full_short = full[impt_cols_list]

# %% Keep only ZWARNING == 0
full_zok = full_short[full_short['ZWARNING'] == 0]

# %%
full_zok.write('full_zok_w1234.fits', format = 'fits')
# %% Plot position in sky
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in full_short['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
ax.set_rlabel_position(120)


cm = plt.cm.get_cmap('RdYlBu_r')
SC = ax.scatter(ra_rad, full_short['DEC'], marker='o', c=full_short['Z'], s=10, lw=0., cmap=cm)
cbar = plt.colorbar(SC, shrink=1., pad=0.05)
cbar.ax.tick_params(labelsize=8)
cbar.set_label('colorbar', fontsize=8)

plt.show()
# %% Save position in sky figure
#plt.savefig('ra_dec_plot.png', dpi=300)

# %% Plot colour distribution g-r
plt.figure()
plt.plot(full_short['Z'],-2.5*np.log10(full_short['FLUX_R']/full_short['FLUX_G']), 'o')
plt.xlabel('Redshift z')
plt.ylabel('g-r')
plt.show()

# %% Plot colour distribution for galaxies
galaxy = full_short['CLASS'] == 'GALAXY'
full_short_g = full_short[galaxy]
plt.figure()
plt.plot(full_short_g['Z'],-2.5*np.log10(full_short_g['FLUX_R']/full_short_g['FLUX_G']), 'o')
plt.show()

# %% Plot position in sky (without colorbar)
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in photo['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
ax.set_rlabel_position(120)

SC = ax.scatter(ra_rad, photo['DEC'], marker='o')
                
# cm = plt.cm.get_cmap('RdYlBu_r')
# SC = ax.scatter(ra_rad, photo['DEC'], marker='o', c=photo['Z'], s=10, lw=0., cmap=cm)
# cbar = plt.colorbar(SC, shrink=1., pad=0.05)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_label('colorbar', fontsize=8)

plt.show()

# %%
photo_wmatch = photo[photo['OBJID'] != -1]

# %% Plot just photo data (with OBJID != -1)
# Code from https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

ra_rad = [x / 180.0 * np.pi for x in photo_wmatch['RA']]

# Make plot.
fig= plt.figure()
ax = plt.subplot(polar=True)

# Set x,y ticks
ax.set_rlabel_position(120)

SC = ax.scatter(ra_rad, photo_wmatch['DEC'], marker='o')
                
# cm = plt.cm.get_cmap('RdYlBu_r')
# SC = ax.scatter(ra_rad, photo['DEC'], marker='o', c=photo['Z'], s=10, lw=0., cmap=cm)
# cbar = plt.colorbar(SC, shrink=1., pad=0.05)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_label('colorbar', fontsize=8)

plt.show()

# %% Plot colour distribution g-r
plt.figure()
plt.plot(full_zok['Z'],-2.5*np.log10(full_zok['FLUX_R']/full_zok['FLUX_G']), 'o')
plt.xlabel('Redshift z')
plt.ylabel('g-r')
plt.show()

# %% Plot colour distribution r-z
plt.figure()
plt.plot(full_zok['Z'],-2.5*np.log10(full_zok['FLUX_Z']/full_zok['FLUX_R']), 'o')
plt.xlabel('Redshift z')
plt.ylabel('r-z')
plt.show()

# %% Filter galaxies
galaxy = full_zok['CLASS'] == 'GALAXY'
full_zok_galaxy = full_zok[galaxy]

# %% Plot colour distribution r-z for galaxies
plt.figure()
plt.plot(full_zok_galaxy['Z'],-2.5*np.log10(full_zok_galaxy['FLUX_Z']/full_zok_galaxy['FLUX_R']), 'o')
plt.title('r-z Distribution for Galaxies')
plt.xlabel('Redshift z')
plt.ylabel('r-z')
plt.show()

# %% Plot colour distribution g-r for galaxies
plt.figure()
plt.plot(full_zok_galaxy['Z'],-2.5*np.log10(full_zok_galaxy['FLUX_R']/full_zok_galaxy['FLUX_G']), 'o')
plt.title('g-r Distribution for Galaxies')
plt.xlabel('Redshift z')
plt.ylabel('g-r')
plt.show()

# %% Filter AGN
agn = full_zok_galaxy['SUBCLASS'] == 'AGN                  '
full_zok_agn = full_zok_galaxy[agn]

# %% Plot colour distribution r-z for AGN
plt.figure()
plt.plot(full_zok_agn['Z'],-2.5*np.log10(full_zok_agn['FLUX_Z']/full_zok_agn['FLUX_R']), 'o')
plt.title('r-z Distribution for AGN Galaxies')
plt.xlabel('Redshift z')
plt.ylabel('r-z')
plt.show()

# %% Plot colour distribution g-r for AGN
plt.figure()
plt.plot(full_zok_agn['Z'],-2.5*np.log10(full_zok_agn['FLUX_R']/full_zok_agn['FLUX_G']), 'o')
plt.title('g-r Distribution for AGN Galaxies')
plt.xlabel('Redshift z')
plt.ylabel('g-r')
plt.show()