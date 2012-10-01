#!/usr/bin/env python
#coding=utf-8


import os
import sys
from sys import *
from os import *
import os.path as op
from numpy import *
from pylab import *
import re


sys.path.append('/net/scratch2/gp/u25400/lasse/ekofisk/svn_py')

from rw_coords_selby_config import distance_azi_backazi,rw_coo_conf_file
import selbyam_tools0_09 as s_amt9
reload(s_amt9)

workdir  = op.abspath(op.realpath('/net/scratch2/gp/u25400/lasse/ekofisk/data/ascii_data_rtz'))
coord_fn = 'stations.dat'
co_file  = op.abspath(op.realpath(op.join(workdir,coord_fn)))
co_array = loadtxt(co_file, dtype = {'names':('num', 'name', 'lat', 'lon'), 'formats':('i','S5','f','f')} ,comments ='#')
co_dict = {}
for ii in co_array:
    stationname = ii[1].upper()
    co = [float(ii[2]),float(ii[3])]
    co_dict[stationname] = co
source_coords = [56.567,3.179]

    
search_string = re.compile(r'(DISPL)\.(?P<stat>[a-zA-Z]+)\.(?P<chan>[A-Z]{3})(?!\d)')

to_rot_dict = {}

for root, dirs, files in os.walk(workdir):
    for name in files:
        fn = op.join(root,name)
        kk = search_string.search(name)
        if kk:
            
            kk_dict = kk.groupdict()
            station = str(kk_dict['stat'].upper())
            channel = str(kk_dict['chan'].upper())
            comp    = channel[-1]

            if comp == 'Z':
                continue
            else:
                if to_rot_dict.has_key(station):
                    templist = to_rot_dict[station]
                else:
                    templist = []
                templist.append(fn)
                to_rot_dict[station] = templist

for key in to_rot_dict:
    templist = to_rot_dict[key]
    if  len(templist) < 2:
        continue
    for kk in templist:
        if kk[-1] == 'T':
            fn1 = kk
        if kk[-1] == 'R':
            fn2 = kk

        
    T_array = loadtxt(fn1)
    R_array = loadtxt(fn2)
    
    time_x   = T_array[:,0]
    time_y   = R_array[:,0]
    problem = 0
    try:
        testsum  = sum(time_x-time_y)
        #print testsum
        if not testsum == 0:
            problem = 1
    except:
        length_diff = len(time_x)-len(time_y)
        #print length_diff
        if not length_diff == 0:
            print 'different length of traces in station %s' %(key)
            problem = 1

    if not problem == 0:
        print 'different time axes for station %s - correcting !' %(key)

        if time_x[0] < time_y[0]:
            start_idx = (abs(time_x- time_y[0])).argmin()
            T_array = T_array[start_idx:]
        if time_x[0] > time_y[0]:
            start_idx = (abs(time_x[0]- time_y)).argmin()
            R_array = R_array[start_idx:]

        if len(T_array) > len(R_array):
            T_array = T_array[:len(R_array)]
        if len(T_array) < len(R_array):
            R_array = R_array[:len(T_array)]
            
    E_array = T_array
    N_array = R_array
    
    timeaxis = time_x

    data_x   = T_array[:,1]
    data_y   = R_array[:,1]

    try:
        coords   = co_dict[key]
    except:
        print 'coordinates for station %s not found'%(key)
        raise SystemExit

    dist,azi,ba    = distance_azi_backazi(source_coords,coords)
    angle = (ba-180.)%360

    try:
        data_e,data_n = s_amt9.rotate_traces(data_x,data_y,angle)
 
        E_array[:,1] = data_e
        N_array[:,1] = data_n

        fn_out_n = 'DISPL.%s.BHN'%(key.upper())
        f_out_n  =  op.abspath(op.realpath(op.join(workdir,fn_out_n)))
        savetxt(f_out_n,N_array)
        fn_out_e = 'DISPL.%s.BHE'%(key.upper())
        f_out_e  =  op.abspath(op.realpath(op.join(workdir,fn_out_e)))
        savetxt(f_out_e,E_array)
        print 'written %s and %s - backazimuth %f degrees (azi %f ) rotated for %f degrees \n'%(fn_out_n,fn_out_e, ba,azi, angle)
    except:
        print 'error in station %s '%(key)
        raise SystemExit
