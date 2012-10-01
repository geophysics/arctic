#!/usr/bin/env python
#coding=utf-8


from numpy import *
from os import *
from os.path import *
import pylab


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import Beach



fn_in = 'gridpoints_erebus_topo_9x9_1000m.txt'
gp_co = loadtxt(fn_in)
grid_lats = gp_co[:81,0]
grid_lons = gp_co[:81,1]

#------

dem_file = 'erebus_dem.txt'
dem = loadtxt(dem_file,skiprows=6)

latmin = -77.6609167
latmax = -77.3919444

lonmin = 166.579250
lonmax = 167.837611

#------


lats = linspace(latmax,latmin,dem.shape[0])
lons = linspace(lonmin,lonmax,dem.shape[1])


latmin_map = -77.58
latmax_map = -77.46

lonmin_map = 166.8
lonmax_map = 167.4
#------

m = Basemap(projection='merc', lon_0=167.15, lat_0=-77.5269444, resolution="h",
            llcrnrlon=lonmin_map, llcrnrlat=latmin_map, urcrnrlon=lonmax_map, urcrnrlat=latmax_map)


# create grids and compute map projection coordinates for lon/lat grid
x, y = m(*meshgrid(lons,lats))

#------

plotfig = plt.figure(97,figsize=(10,10) )
#plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)
# ensure that the rc commands come before the plot command!
pylab.rc("axes", linewidth=2.0)
pylab.rc("lines", markeredgewidth=2.0)  # this also grows the points/lines in the plot..

#------


# Make contour plot
cs = m.contourf(x, y, dem, 100)#, colors="k", lw=0.5, alpha=0.3)
cs = m.contour(x, y, dem, 16, colors="k", lw=0.4, alpha=0.3)
plt.clabel(cs,cs.levels[1::2],inline=True,fmt='%i',fontsize=15)

#------

# update the font size of the x and y axes
fontsize=26
ax = pylab.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
#------


#m.drawcountries(color="red", linewidth=1)
#plt.colorbar()


# Draw a lon/lat grid (20 lines for an interval of one degree)
m.drawparallels(linspace(latmin-.1, latmax+.1, 16), labels=[1,1,0,0], fmt="%.2f", dashes=[2,2])
m.drawmeridians(linspace(lonmin-.1, lonmax+.1, 12), labels=[0,0,1,1], fmt="%.2f", dashes=[2,2])
#------

#m.fillcontinents()

#m.drawmapscale(lon=166.89,lat=-77.47,length=5,units='km',lon0=166.8,lat0=-77.45,barstyle='fancy',fontsize=11,labelstyle='simple',yoffset=0.1)

# Plot station positions and names into the map
# again we have to compute the projection of our lon/lat values
#------

recs_lats = [-77.534643,-77.530420,-77.531603,-77.510599,-77.521976,-77.528559]
recs_lons = [167.084940,167.139711,166.932620,167.142086,167.147465,167.170830]

names = [" CON", " E1S", " HOO", " LEH",' NKB',' RAY']
x, y = m(recs_lons, recs_lats)
m.scatter(x, y, 200, color="b", marker="v",lw=2, edgecolor="k", zorder=3)
for i in range(len(names)):
    plt.text(x[i], y[i], names[i], va="top",ha='left',linespacing=0.4, family="monospace", weight="bold", zorder=4,fontsize=15)

#------
#x, y = m(grid_lons, grid_lats)
#m.scatter(x, y, 200, color="g", marker="o",lw=1, edgecolor="b", zorder=3)
#------

x,y =m([167.15],[-77.526944])
m.scatter(x,y,300,color='y',marker='x',lw=8,edgecolor='y',zorder=2)
plt.text(x[0],y[0],' ',va='baseline',ha='right',family="monospace", weight="bold")


for i in ['eps','pdf','svg','png','ps']:
    outname = 'setup_erebus_local.%s'%(i)
    plt.savefig(outname)

plt.show()

plt.close('all')
#-----------------------------------------------------------------------------------

dem_file = 'erebus_dem.txt'

dem = loadtxt(dem_file,skiprows=6)

latmin = -77.6609167
latmax = -77.3919444

lonmin = 166.579250
lonmax = 167.837611

lats = linspace(latmax,latmin,dem.shape[0])
lons = linspace(lonmin,lonmax,dem.shape[1])


latsrange = grid_lats.max()-grid_lats.min()
lonsrange = grid_lons.max()-grid_lons.min()


plotfig = plt.figure(98,figsize=(10,10) )
latmin_map = grid_lats.min() - .2*latsrange
latmax_map = grid_lats.max() + .2*latsrange
lonmin_map = grid_lons.min() - .2*lonsrange
lonmax_map = grid_lons.max() + .2*lonsrange


# latmin_map = -77.58
# latmax_map = -77.46

# lonmin_map = 166.8
# lonmax_map = 167.4

m = Basemap(projection='merc', lon_0=167.15, lat_0=-77.5269444, resolution="h",
            llcrnrlon=lonmin_map, llcrnrlat=latmin_map, urcrnrlon=lonmax_map, urcrnrlat=latmax_map)


# create grids and compute map projection coordinates for lon/lat grid
x, y = m(*meshgrid(lons,lats))

#plotfig = plt.figure(98,figsize=(10,10) )
#plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)
# ensure that the rc commands come before the plot command!
pylab.rc("axes", linewidth=2.0)
pylab.rc("lines", markeredgewidth=2.0)  # this also grows the points/lines in the plot..



# Make contour plot
cs = m.contourf(x, y, dem, 100)#, colors="k", lw=0.5, alpha=0.3)
cs = m.contour(x, y, dem, 16, colors="k", lw=0.4, alpha=0.3)

# Draw a lon/lat grid (20 lines for an interval of one degree)
m.drawparallels(linspace(latmin_map-.1, latmax_map+.1, 14), labels=[1,1,0,0], fmt="%.2f", dashes=[2,2])
m.drawmeridians(linspace(lonmin_map-.1, lonmax_map+.1, 12), labels=[0,0,1,1], fmt="%.2f", dashes=[2,2])

x, y = m(grid_lons, grid_lats)
m.scatter(x, y, 200, color="b", marker="o",lw=2, edgecolor="b", zorder=3)

x,y =m([167.15],[-77.526944])
m.scatter(x,y,300,color='y',marker='x',lw=5,edgecolor='y',zorder=2)


for i in ['eps','pdf','svg','png','ps']:
    outname = 'setup_erebus_grid.%s'%(i)
    plt.savefig(outname)
plt.ion()
plt.show()


plt.close('all')
