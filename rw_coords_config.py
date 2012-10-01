#!/usr/bin/env python


import os
from sys import *
from os import *
import math
import cmath
import re
import scipy
import string
import numpy
from numpy import *
from scipy.io import read_array, write_array
from pylab import *
import shutil
import ConfigParser
from scipy.linalg import *
from scipy import io  as s_io
import random as rdm
from pylab import *
import pickle as pp
from scipy.integrate import simps as simps

import  Scientific.IO.NetCDF as sioncdf
import time
    
from lxml import etree as xet
import pymseed
from subprocess import *


import pickle as pp
import numpy.random as Rand  
import time


R      =   6371000.
rad_to_deg  = 180./pi


def rw_coo_conf_file(filename):

    print 'reading file "%s"'%(filename)
    ddd = loadtxt(filename, dtype = {'names':('num', 'name', 'lat', 'lon'), 'formats':('i','S5','f','f')} ,comments ='#')
    print '...DONE \n'
    stationlist = []
    for ii in arange(len(ddd)):
        stationlist.append(ddd[ii][1])
    
    outfilename      = 'conf_file_suited_dist400km'+filename
    outfile_weights  = 'station_weights'+filename

    rec_co_file    = file(outfilename,'w')
    weights_file   = file(outfile_weights,'w')

    stat_co_conf   = ConfigParser.ConfigParser()

    stations_used = 0
    #source     = [53.01,9.63]
    source     = [53.01,9.63]
    

    lo_stats = []
    for idx,entry in enumerate(ddd):
        lo_stats.append(entry[1])

    stations_order = argsort(lo_stats)


    for station_index in stations_order:
        stat_info = ddd[station_index]
        
        stationname = stat_info[1]
        lat = stat_info[2]
        lon = stat_info[3]
            
        rec_co =[lat,lon]
        dist,dist_aa,distance_in_degree = distance(rec_co,source)

        
        if not 60 < dist < 400000:
            continue

        stations_used += 1
        stat_idx       = stations_used
        
        print  stationname,  int(dist/1000.),' km' , '(', dist_aa ,' km) -- (%i)'%(station_index+1),'station index: ',stat_idx 
        stat_co_conf.add_section(stationname)
        stat_co_conf.set(stationname,'index',str(stat_idx))
        stat_co_conf.set(stationname,'lat',str(lat))

        stat_co_conf.set(stationname,'lon',str(lon))

        weights_string = '%s \t %f'%(stationname,1.)
        weights_file.write(weights_string)


    
    stat_co_conf.write(rec_co_file)
    rec_co_file.close()
    weights_file.close()
    
    print '%i stations'%(stations_used)
#----------------------------------------------------------------------
def distance(coord_s,coord_r):
    """
    Calculating the (back-)azimuth (clockwise) and the distance between the
    surface-projection of the source and the receiver.

    Input:
    -- array with source coordinates in (lat,lon,depth)
    -- array with receiver coordinates in (lat,lon,depth)
    -- Configuration-dicotionary

    Output:
    -- (back-)azimuth (in degree) -- as seen from 2nd argument (receiver)
    -- distance between surface-projection of the source and receiver (in m)
    """

    #print  'calculating distance...'
    args =[]
    cmd = 'geod'
    args.append(cmd)
    cmd2 = '+ellps=WGS84'
    cmd3 = '+units=km'
    cmd4 = '-I'
    cmd5 = '-p'
    args.append(cmd2)
    args.append(cmd3)
    args.append(cmd4)
    args.append(cmd5)


    

    lat1  =  float(coord_s[0])/rad_to_deg
    lon1  =  float(coord_s[1])/rad_to_deg
    lat2  =  float(coord_r[0])/rad_to_deg
    lon2  =  float(coord_r[1])/rad_to_deg
    
    arg_str = '%.4f %.4f  %.4f  %.4f\n'%(coord_s[0],coord_s[1],coord_r[0],coord_r[1])
    
    aa = Popen(args,stdout=PIPE,stdin=PIPE).communicate(arg_str)[0]

    
    distance_in_m      = R * arccos( sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1) )
    distance_in_degree = distance_in_m/R


    return  distance_in_m,aa.split()[2],distance_in_degree
 #----------------------------------------------------------------------
def calc_azi(s_lat,s_lon,r_lat,r_lon,sr_dist):
        
    if sin(r_lon-s_lon) < 0:       
        tc1=arccos((sin(r_lat)-sin(s_lat)*cos(sr_dist))/(sin(sr_dist)*cos(s_lat)))    
    else:       
        tc1=2*pi-arccos((sin(r_lat)-sin(s_lat)*cos(sr_dist))/(sin(sr_dist)*cos(s_lat)))  

            #atan2(  	sin(?long).cos(lat2),
            #cos(lat1).sin(lat2) ? sin(lat1).cos(lat2).cos(?long) )

        if (cos(s_lat) < 0.000001) or (cos(r_lat) < 0.000001):
            if (s_lat > 0):
                tc1 = pi # starting from N pole
            elif (s_lat < 0):
                tc1 = 0.   #  starting from S pole

    azi = -tc1*rad_to_deg%360
      
    return azi

 #----------------------------------------------------------------------
def azi_backazi(coord_s,coord_r):     
 
    rad_to_deg  = 180./pi

    lat1  =  float(coord_s[0])/rad_to_deg
    lon1  =  float(coord_s[1])/rad_to_deg
    lat2  =  float(coord_r[0])/rad_to_deg
    lon2  =  float(coord_r[1])/rad_to_deg
  
    dummy1,dummy2,distance_in_degree = distance(coord_s,coord_r) 
    azimuth_s_r        = calc_azi(lat1,lon1,lat2,lon2,distance_in_degree)
    back_azimuth_r_s   = calc_azi(lat2,lon2,lat1,lon1,distance_in_degree)

    #ba_d    = int(back_azimuth_r_s)
    #ba_m    = (back_azimuth_r_s-ba_d)*60.
    #ba_fm   = int(ba_m)
    #ba_s    =  (ba_m-ba_fm)*60
    #ba_dms  = [ba_d,ba_fm,ba_s]
    #a_d     = int(azimuth_s_r)
    #a_m     = (azimuth_s_r-a_d)*60.
    #a_fm    = int(a_m)
    #a_s     =  (a_m-a_fm)*60
    #a_dms   = [a_d,a_fm,a_s]
    
    #print aa
    #print ba_dms, '(',back_azimuth_r_s,')'
    #print a_dms, '(',azimuth_s_r,')'
    #print [back_azimuth_r_s,azimuth_s_r],distance_in_m/1000,'\n'
    #print 'distance tool, own:',aa,distance_in_m/1000,'\n'



    return azimuth_s_r, back_azimuth_r_s 
 
 #---------------------------------------------------------------------- 
  
        
if __name__ == '__main__':      
        
    ff = str(sys.argv[1])
    rw_coo_conf_file(ff)    

    
    
    
    
    
    
    
